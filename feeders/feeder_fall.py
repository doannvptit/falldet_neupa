#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feeder for NTU120 binary fall detection (fine-tune of NeuroPath).

Data: data/ntu120_fall_v2/{x_train,x_test,y_train,y_test}.npy produced by
data_gen/prepare_ntu120_fall_v2.py. x arrays (N, T_CAP=150, 2, 25, 3) float16
mmap-loaded (per-MS-G3D pre-normalized); y arrays are binary uint8
(1 = falling / A43, 0 = not falling).

Subsampling: to fight the extreme class imbalance at train-time we keep all
falling samples and only `subsample_ratio * n_falling` non-falling samples.
The test loader keeps a seeded, FIXED subset; the train loader re-draws its
random negatives each epoch (see resample_indices). Use subsample_ratio<=0
to keep the whole (realistic) distribution.
"""
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train',
                 random_choose=False, random_shift=False, random_move=False,
                 random_rot=False, window_size=-1, normalization=False, debug=False,
                 use_mmap=True, bone=False, vel=False, multi_input=True,
                 subsample_ratio=8, sample_seed=1024, zero_z=False, projection_path=None,
                 image2d_multiview=False, eval_views=('overhead', 'side_left', 'side_right')):
        self.debug = debug
        self.data_path = data_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.multi_input = multi_input
        self.subsample_ratio = subsample_ratio
        self.sample_seed = sample_seed
        self.epoch_extra = 0
        self.zero_z = zero_z
        self.image2d_multiview = image2d_multiview
        self.eval_views = tuple(eval_views)
        if image2d_multiview and (not zero_z or projection_path or not self.eval_views):
            raise ValueError('Multiview image XY requires zero_z, eval views and no PCA')
        self.projection = None
        if projection_path:
            if not zero_z:
                raise ValueError('PCA 2D projection requires zero_z=True')
            from pose_projection import load_projection
            self.projection = load_projection(projection_path)
        self.load_data()

    def load_data(self):
        if self.image2d_multiview:
            with open(os.path.join(self.data_path, 'stats.json')) as f:
                if json.load(f).get('coordinate_mode') != 'camera_xyz':
                    raise ValueError('Virtual camera training requires camera_xyz data without body rotations')
        self.x_path = os.path.join(self.data_path, f'x_{self.split}.npy')
        self.y_path = os.path.join(self.data_path, f'y_{self.split}.npy')
        # memmap: O(1) RAM, read-on-demand (data stays on disk)
        self.data = np.load(self.x_path, mmap_mode='r')        # (N, T, M, V, C)
        self.label_full = np.load(self.y_path).astype(np.int64)
        self._apply_selection()

        if self.debug:
            limit = min(64, len(self.indices))
            pos_dbg = self.indices[self.label == 1][:limit // 4]
            neg_dbg = self.indices[self.label == 0][:limit - len(pos_dbg)]
            self.indices = np.sort(np.concatenate([pos_dbg, neg_dbg]).astype(np.int64))
            self.label = self.label_full[self.indices]
            self.sample_name = [f'{self.split}_{int(i)}' for i in self.indices]

    def _apply_selection(self):
        # classes: falling=1, not falling=0
        pos = np.where(self.label_full == 1)[0]
        neg = np.where(self.label_full == 0)[0]
        if self.subsample_ratio and self.subsample_ratio > 0 and len(neg) > len(pos):
            n_keep = min(len(neg), len(pos) * int(self.subsample_ratio))
            seed = self.sample_seed + {'train': 0, 'test': 1}[self.split] + self.epoch_extra
            rng = np.random.RandomState(seed)
            neg_sel = np.sort(rng.choice(neg, size=n_keep, replace=False))
            indices = np.sort(np.concatenate([pos, neg_sel]))
        else:
            indices = np.sort(np.concatenate([pos, neg]))
        self.indices = indices.astype(np.int64)
        self.label = self.label_full[self.indices]
        self.sample_name = [f'{self.split}_{int(i)}' for i in self.indices]
        if self.image2d_multiview and self.split != 'train':
            self.indices = np.repeat(self.indices, len(self.eval_views))
            self.label = self.label_full[self.indices]
            self.sample_name = [f'{self.split}_{int(i)}_{self.eval_views[k % len(self.eval_views)]}'
                                for k, i in enumerate(self.indices)]

    def resample_indices(self, epoch):
        """Called by the Processor each training epoch: re-draws the negative
        subsample (same size) so every epoch sees different non-fall sequences."""
        if self.split == 'train' and self.subsample_ratio and self.subsample_ratio > 0:
            self.epoch_extra = int(epoch) + 1
            self._apply_selection()

    def set_epoch(self, epoch):
        self.resample_indices(epoch)

    def get_mean_map(self):          # kept for interface compatibility
        pass

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        idx = self.indices[index]
        label = int(self.label[index])
        data_numpy = np.array(self.data[idx], dtype=np.float32)   # (T, M, V, C)
        if self.image2d_multiview:
            from pose_2d import virtual_camera
            view = None if self.split == 'train' else self.eval_views[index % len(self.eval_views)]
            data_numpy = virtual_camera(data_numpy, view=view)
        if self.projection is not None:
            from pose_projection import project_pose
            data_numpy = project_pose(data_numpy, self.projection)
        data_numpy = data_numpy.transpose(3, 0, 2, 1)             # -> (C, T, V, M)

        if self.zero_z:
            data_numpy[2] = 0

        valid_frame_num = np.sum(np.abs(data_numpy).sum(axis=(0, 2, 3)) > 0) if self.zero_z else np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num,
                                             self.p_interval, self.window_size)
        if self.random_rot:
            if self.zero_z:
                angle = np.random.uniform(-0.3, 0.3)
                c, s = np.cos(angle), np.sin(angle)
                data_numpy[:2] = np.einsum('ij,jtvm->itvm', np.array([[c, -s], [s, c]]), data_numpy[:2])
            else:
                data_numpy = tools.random_rot(data_numpy)
                data_numpy = data_numpy.numpy()
        if self.zero_z:
            data_numpy[2] = 0

        if self.image2d_multiview:
            from pose_2d import normalize_image_xy, augment_image_xy
            data_numpy = normalize_image_xy(data_numpy)
            if self.split == 'train':
                data_numpy = augment_image_xy(data_numpy)

        data_numpy_1 = data_numpy.copy()
        if self.bone:
            from .bone_pairs import ntu_pairs
            for v1, v2 in ntu_pairs:
                data_numpy_1[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
        if self.vel:
            data_numpy_1[:, :-1] = data_numpy_1[:, 1:] - data_numpy_1[:, :-1]
            data_numpy_1[:, -1] = 0

        data_numpy_2 = data_numpy_1.copy()
        if not self.vel:
            data_numpy_2[:, :-1] = data_numpy_1[:, 1:] - data_numpy_1[:, :-1]
            data_numpy_2[:, -1] = 0
        elif self.bone:
            data_numpy_2 = data_numpy.copy()
        else:
            from .bone_pairs import ntu_pairs
            for v1, v2 in ntu_pairs:
                data_numpy_2[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]

        if self.multi_input:
            data_numpy_1 = torch.from_numpy(data_numpy_1).unsqueeze(0)
            data_numpy_2 = torch.from_numpy(data_numpy_2).unsqueeze(0)
            return torch.cat([data_numpy_1, data_numpy_2], dim=0), label, index
        return torch.from_numpy(data_numpy_1), label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
