from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
import supervision as sv

from .detection import Detection

@dataclass
class TrackedObject:
    frame_idx: int
    track_id: int
    cls: str
    class_id: int
    xyxy: Tuple[int, int, int, int]
    score: float

class Tracker:
    def __init__(self, fps: float = 30.0):
        # Initialize ByteTrack
        # frame_rate is helpful for internal Kalman filter tuning
        self.tracker = sv.ByteTrack(frame_rate=fps)

    def update(self, frame_idx: int, detections: List[Detection]) -> List[TrackedObject]:
        """
        Update the tracker with detections from a single frame.
        Returns the active tracks for this frame.
        """
        if not detections:
            # Even if empty, we must call update so internal state decays
            # sv.Detections.empty() is not directly available, so we make one
            sv_detections = sv.Detections(
                xyxy=np.empty((0, 4)),
                class_id=np.array([]),
                confidence=np.array([])
            )
        else:
            # Convert to supervision Detections
            xyxy = np.array([d.xyxy for d in detections])
            class_id = np.array([d.class_id for d in detections])
            confidence = np.array([d.score for d in detections])
            
            # We also want to preserve the original class name (str) for later
            # sv.Detections doesn't natively track custom metadata per box easily in update()
            # but we can map class_id back to name if needed, or rely on the input list order?
            # ByteTrack reorders/filters, so we need a way to persist class names.
            # A common trick is to map class_id -> name globally if 1:1.
            # Here we'll construct a lookup for the current frame's detections.
            class_id_to_name = {d.class_id: d.cls for d in detections}

            sv_detections = sv.Detections(
                xyxy=xyxy,
                class_id=class_id,
                confidence=confidence
            )

        # Run ByteTrack update
        tracked_detections = self.tracker.update_with_detections(sv_detections)

        # Convert back to our TrackedObject
        results = []
        
        # Iterate over the returned supervision detections which now have tracker_id
        for i in range(len(tracked_detections)):
            t_xyxy = tracked_detections.xyxy[i]
            t_class_id = tracked_detections.class_id[i]
            t_confidence = tracked_detections.confidence[i]
            t_track_id = tracked_detections.tracker_id[i]
            
            # Lookup class name - note: ByteTrack might return an object from a previous frame
            # whose class_id might not be in the *current* frame's input list if strictly occluded?
            # Actually ByteTrack returns matched detections from *current* frame input mostly.
            # If it's a "prediction" without a match, ByteTrack typically doesn't output it in result 
            # unless configured. supervision's update_with_detections returns matched detections.
            
            # If class_id mapping is simple (static), we are good.
            # If dynamic, we might default to "unknown" or try to recover.
            # For now, we assume we can just reuse the class_id to infer type (e.g. 0=ball, 1=player)
            # BUT we don't have the static map here yet. 
            # Let's assume we can pass the name through if we had a global map. 
            # For this specific frame, we can try:
            cls_name = class_id_to_name.get(t_class_id, str(t_class_id))

            obj = TrackedObject(
                frame_idx=frame_idx,
                track_id=int(t_track_id),
                cls=cls_name,
                class_id=int(t_class_id),
                xyxy=tuple(map(int, t_xyxy)),
                score=float(t_confidence)
            )
            results.append(obj)
            
        return results

def track_segment(detections: List[Detection], fps: float = 30.0) -> List[TrackedObject]:
    """
    Batch process a list of detections (which might span multiple frames).
    Returns a flat list of tracked objects.
    """
    # 1. Group detections by frame
    dets_by_frame = {}
    max_frame = 0
    for d in detections:
        dets_by_frame.setdefault(d.frame_idx, []).append(d)
        max_frame = max(max_frame, d.frame_idx)
    
    # 2. Initialize tracker
    tracker = Tracker(fps=fps)
    all_tracks = []
    
    # 3. Iterate frame by frame
    # We must go sequentially from min to max frame to maintain state
    min_frame = min(dets_by_frame.keys()) if dets_by_frame else 0
    
    for f_idx in range(min_frame, max_frame + 1):
        frame_dets = dets_by_frame.get(f_idx, [])
        tracks = tracker.update(f_idx, frame_dets)
        all_tracks.extend(tracks)
        
    return all_tracks

