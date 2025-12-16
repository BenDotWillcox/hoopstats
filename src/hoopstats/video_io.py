from pathlib import Path
from typing import Iterator, Tuple, Optional, Generator
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float


class VideoLoader:
    """
    Handles loading and iterating through video frames.
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[VideoInfo] = None

    def _ensure_open(self):
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video: {self.path}")

    def get_info(self) -> VideoInfo:
        """
        Retrieve video metadata.
        """
        self._ensure_open()
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        duration = total_frames / fps if fps > 0 else 0.0
        
        self._info = VideoInfo(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration
        )
        return self._info

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yields (frame_index, frame_array) for the entire video.
        """
        self._ensure_open()
        # Always reset to beginning for a fresh iteration
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    def close(self):
        """
        Release video resources.
        """
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
