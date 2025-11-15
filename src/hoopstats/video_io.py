from pathlib import Path
from typing import Iterator, Tuple, Any

import cv2
import numpy as np


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def segment_video(cap: cv2.VideoCapture) -> list[dict]:
    """
    Return a list of segments.
    For now you can do fixed segments, e.g. every N seconds:
      [{ "start_frame": 0, "end_frame": 5000, "camera_id": 0 }, ...]
    Replace with real scene-cut / quarter detection later.
    """
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_per_segment = int(10 * fps)  # 10-second chunks to start

    segments = []
    start = 0
    while start < frame_count:
        end = min(start + frames_per_segment, frame_count)
        segments.append(
            {"start_frame": start, "end_frame": end, "camera_id": 0}
        )
        start = end

    return segments


def iter_segment_frames(cap: cv2.VideoCapture, segment: dict) -> Iterator[Tuple[int, np.ndarray]]:
    start = segment["start_frame"]
    end = segment["end_frame"]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_idx = start
    while frame_idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame_idx, frame
        frame_idx += 1
