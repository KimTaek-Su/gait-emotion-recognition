from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np


class BasePoseBackend(ABC):
    @abstractmethod
    def extract(self, frame) -> Optional[list[list[float]]]:
        """
        Return keypoints for one frame as [[x, y, z], ...]
        or None if detection fails.
        """
        raise NotImplementedError


class DummyPoseBackend(BasePoseBackend):
    def extract(self, frame):
        return None


class MediaPipePoseBackend(BasePoseBackend):
    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError(
                "mediapipe is not installed. Install it or choose another pose backend."
            ) from e

        self.mp = mp
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame) -> Optional[list[list[float]]]:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)

            if not result.pose_landmarks:
                return None

            keypoints = []
            for lm in result.pose_landmarks.landmark:
                keypoints.append([float(lm.x), float(lm.y), float(lm.z)])

            return keypoints
        except Exception:
            return None

    def __del__(self):
        try:
            self.pose.close()
        except Exception:
            pass


def create_pose_backend(name: str) -> BasePoseBackend:
    name = name.lower()

    if name == "mediapipe":
        return MediaPipePoseBackend()

    if name == "yolo_pose":
        # TODO:
        # Replace with actual YOLO pose implementation later.
        # For now we intentionally raise a clear error so that
        # users do not think YOLO is already working.
        raise NotImplementedError(
            "yolo_pose backend is not implemented yet. "
            "Use 'mediapipe' for now, or implement YOLO pose here."
        )

    if name == "dummy":
        return DummyPoseBackend()

    raise ValueError(f"Unsupported pose backend: {name}")