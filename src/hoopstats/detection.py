from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    frame_idx: int
    cls: str
    xyxy: Tuple[int, int, int, int]
    score: float


# TODO: load your RF-DETR / YOLO model here once, at module import or via init
_model = None


def load_model_if_needed():
    global _model
    if _model is None:
        # _model = ...
        pass


def run_detection_on_segment(frames) -> List[Detection]:
    """
    frames: iterator of (frame_idx, frame)
    returns: flat list of Detection objects
    """
    load_model_if_needed()
    all_dets: List[Detection] = []

    for frame_idx, frame in frames:
        # TODO: run your detector here, convert outputs to Detection instances
        # Example:
        # preds = _model(frame)
        # for p in preds:
        #     all_dets.append(Detection(frame_idx, p.cls, p.xyxy, p.score))
        pass

    return all_dets
