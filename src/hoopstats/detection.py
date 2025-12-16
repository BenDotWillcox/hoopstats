from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from inference import get_model
import supervision as sv
import cv2

from .config import ROBOFLOW_API_KEY


@dataclass
class Detection:
    frame_idx: int
    cls: str
    class_id: int
    xyxy: Tuple[int, int, int, int]
    score: float


# Model configuration from the notebook
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
PLAYER_DETECTION_MODEL_CONFIDENCE = 0.4
PLAYER_DETECTION_MODEL_IOU_THRESHOLD = 0.9

# Global model instance
_model = None


def load_model_if_needed():
    global _model
    if _model is None:
        print(f"Loading Roboflow model: {PLAYER_DETECTION_MODEL_ID}...")
        try:
            _model = get_model(
                model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
            print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Roboflow model: {e}")


def run_detection_on_segment(frames) -> List[Detection]:
    """
    Run detection on a segment of frames.

    Args:
        frames: iterator of (frame_idx, frame) tuples

    Returns:
        List of Detection objects
    """
    load_model_if_needed()
    all_dets: List[Detection] = []

    # Mapping from class_id to class_name (approximate based on notebook)
    # The notebook implies these classes:
    # ball, ball-in-basket, number, player, player-in-possession,
    # player-jump-shot, player-layup-dunk, player-shot-block, referee, rim
    # We will let the model provide the class names if possible, or map them later.
    # For now, we store what we get.

    print("Running detection on segment...")
    for frame_idx, frame in frames:
        # Run inference
        # The notebook uses: .infer(frame, confidence=..., iou_threshold=...)
        # The result is a list, we take [0] for the first image
        try:
            result = _model.infer(
                frame,
                confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
                iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD
            )[0]

            # Convert to supervision Detections for easy handling
            detections_sv = sv.Detections.from_inference(result)

            # Iterate through detections
            for i in range(len(detections_sv)):
                xyxy = detections_sv.xyxy[i]
                class_id = detections_sv.class_id[i]
                confidence = detections_sv.confidence[i]

                # Get class name if available in data, otherwise generic
                class_name = detections_sv.data.get(
                    "class_name", ["unknown"] * len(detections_sv))[i]

                det = Detection(
                    frame_idx=frame_idx,
                    cls=str(class_name),
                    class_id=int(class_id),
                    xyxy=tuple(map(int, xyxy)),
                    score=float(confidence)
                )
                all_dets.append(det)

        except Exception as e:
            print(f"Warning: Detection failed for frame {frame_idx}: {e}")
            continue

    print(f"Detection complete. Found {len(all_dets)} objects.")
    return all_dets
