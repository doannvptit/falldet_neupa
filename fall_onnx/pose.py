"""Local YOLOX-s + RTMPose-m Halpe26, all inference through ONNX Runtime."""
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from .pose_helpers import bbox_xyxy2cs, top_down_affine, get_simcc_maximum
from .runtime import create_session

MODELS = Path(__file__).resolve().parents[1]/'models'
DEFAULT_DETECTOR = MODELS/'yolox_s.onnx'
DEFAULT_POSE = MODELS/'rtmpose_m_halpe26.onnx'


def _inference(session, image):
    tensor = np.ascontiguousarray(image.transpose(2,0,1)[None],dtype=np.float32)
    return session.run(None,{session.get_inputs()[0].name:tensor})


class _Detector:
    def __init__(self, path, device, threads, score_thr):
        # Use the reference affine/decode preprocessing, with our own thread-controlled
        # local-only session instead of BaseTool's automatic model downloader.
        self.session = create_session(path,device,threads)
        self.model_input_size = (640,640)
        self.backend, self.det_mode = 'onnxruntime', 'human'
        self.nms_thr, self.score_thr = .45, score_thr

    def postprocess(self, outputs, ratio=1.):
        # The pinned human-only detector includes NMS and returns xyxy+score.
        # Respect our CLI threshold; rtmlib's generic embedded-NMS path uses .3.
        if outputs.shape[-1] == 5:
            detections = outputs[0]
            detections = detections[detections[:,4] >= self.score_thr]
            detections = detections[np.argsort(-detections[:,4],kind='stable')]
            return detections[:,:4]/ratio
        raise ValueError('Expected pinned YOLOX human detector with embedded NMS (N,5)')

    def __call__(self, image: np.ndarray):
        image, ratio = self.preprocess(image)
        outputs = _inference(self.session,image)[0]
        results = self.postprocess(outputs, ratio)
        return results

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for YOLOX model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - padded_img (np.ndarray): Preprocessed image.
            - ratio (float): Scale factor applied to the image.
        """
        if img.shape[:2] == tuple(self.model_input_size[:2]):
            padded_img = img.copy()
            ratio = 1.
        else:
            if len(img.shape) == 3:
                padded_img = np.ones(
                    (self.model_input_size[0], self.model_input_size[1], 3),
                    dtype=np.uint8) * 114
            else:
                padded_img = np.ones(self.model_input_size,
                                     dtype=np.uint8) * 114

            ratio = min(self.model_input_size[0] / img.shape[0],
                        self.model_input_size[1] / img.shape[1])
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
            padded_shape = (int(img.shape[0] * ratio),
                            int(img.shape[1] * ratio))
            padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        return padded_img, ratio


class _Pose:
    def __init__(self, path, device, threads):
        self.session = create_session(path,device,threads)
        self.model_input_size = (192,256)  # W,H (rtmlib convention)
        self.mean = (123.675,116.28,103.53)
        self.std = (58.395,57.12,57.375)
        self.backend, self.to_openpose = 'onnxruntime', False

    def __call__(self, image: np.ndarray, bboxes):
        if len(bboxes) == 0:
            return np.empty((0,26,2),np.float32), np.empty((0,26),np.float32)

        keypoints, scores = [], []
        for bbox in bboxes:
            img, center, scale = self.preprocess(image, bbox)
            outputs = _inference(self.session,img)
            kpts, score = self.postprocess(outputs, center, scale)

            keypoints.append(kpts)
            scores.append(score)

        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)


        return keypoints, scores

    def preprocess(self, img: np.ndarray, bbox: list):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            bbox (list):  xyxy-format bounding box of target.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        bbox = np.array(bbox)

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale,
                                             center, img)
        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            resized_img = (resized_img - self.mean) / self.std

        return resized_img, center, scale

    def postprocess(
            self,
            outputs: List[np.ndarray],
            center: Tuple[int, int],
            scale: Tuple[int, int],
            simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        keypoints = keypoints / self.model_input_size * scale
        keypoints = keypoints + center - scale / 2

        return keypoints, scores


class PoseEstimator:
    def __init__(self, detector=DEFAULT_DETECTOR, pose=DEFAULT_POSE,
                 device='cpu', threads=4, det_confidence=.5, max_people=2):
        if not 0 <= det_confidence <= 1 or max_people < 1:
            raise ValueError('Invalid detection confidence/max_people')
        for path in (detector,pose):
            if not Path(path).is_file():
                raise FileNotFoundError(f'Missing {path}; run scripts/download_pose_models.py')
        self.detector = _Detector(detector,device,threads,det_confidence)
        self.pose = _Pose(pose,device,threads)
        if self.detector.session.get_inputs()[0].shape != [1,3,640,640]:
            raise ValueError('Expected YOLOX-s input [1,3,640,640]')
        if self.pose.session.get_inputs()[0].shape[1:] != [3,256,192]:
            raise ValueError('Expected RTMPose-m Halpe26 input [1,3,256,192]')
        joints = self.pose.session.get_outputs()[0].shape[1]
        if isinstance(joints,int) and joints != 26:
            raise ValueError('Expected 26-keypoint Halpe pose model')
        self.max_people = max_people
        self.providers = dict(detector=self.detector.session.get_providers()[0],
                              pose=self.pose.session.get_providers()[0])

    def __call__(self, bgr):
        boxes = np.asarray(self.detector(bgr),np.float32).reshape(-1,4)
        if not len(boxes):
            # RTMPose otherwise treats an empty list as a full-image bbox.
            return np.empty((0,26,2),np.float32),np.empty((0,26),np.float32),boxes
        valid = np.isfinite(boxes).all(axis=1) & (boxes[:,2]>boxes[:,0]) & (boxes[:,3]>boxes[:,1])
        boxes = boxes[valid][:self.max_people]  # YOLOX NMS is confidence ordered.
        if not len(boxes):
            return np.empty((0,26,2),np.float32),np.empty((0,26),np.float32),boxes
        points,scores = self.pose(bgr,bboxes=boxes)
        if points.shape != (len(boxes),26,2) or scores.shape != (len(boxes),26):
            raise ValueError('Unexpected Halpe26 model output')
        return points.astype(np.float32),scores.astype(np.float32),boxes
