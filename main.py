# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
if hasattr(yaml, 'warnings'):
    yaml.warnings({'YAMLLoadWarning': False})

from parser import Parser
from processor import Processor
from utils import init_seed

if __name__ == '__main__':
    # single-instance guard: one training at a time on this machine (e.g. two
    # agents sharing one repo). The lock is created by the launcher wrapper and
    # removed after the run; a stale lock is cleared explicitly.
    _TRAIN_LOCK = os.path.join('work_dir', '.train.lock')
    if os.path.exists(_TRAIN_LOCK):
        print(f'main.py: {_TRAIN_LOCK} exists -> another training is running or '
              'its lock is stale. Refusing to start a duplicate.', flush=True)
        raise SystemExit(0)

    default_config_file = 'config/nturgbd-cross-subject/test23.yaml'
    parser = Parser(default_config_file)
    args = parser.get_args()
    init_seed(args.seed)

    processor = Processor(args)
    processor.start()