"""
Shot event detection.

Heuristic detector built on the detection model's action classes:

- A *shot attempt* is a run of frames containing a `player-jump-shot` (5) or
  `player-layup-dunk` (6) detection. Consecutive shot-pose frames (with a
  small gap tolerance) are grouped into one attempt window; the release frame
  is the last frame of the window.
- An attempt is a *make* if a `ball-in-basket` (1) detection appears within a
  short window after the attempt starts. Each ball-in-basket group is consumed
  by at most one attempt.
- The *shooter* is the tracked player whose box has the highest IoU with the
  shot-pose box on the release frame; team and jersey number come from the
  track-level maps built earlier in the pipeline.

Court-geometry helpers (`distance_to_nearest_hoop`, `classify_shot_type`) are
pure functions so the 2PT/3PT rule can be unit tested without video.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import ShotEvent
from .detection import Detection

# Detection model class ids (see detection.CLASS_NAMES)
SHOT_ATTEMPT_CLASS_IDS = {5, 6}  # player-jump-shot, player-layup-dunk
MAKE_CLASS_ID = 1                # ball-in-basket
PLAYER_CLASS_IDS = {3, 4, 5, 6, 7}

# NBA court geometry (feet). Court is 94 x 50, hoops centered on y=25.
HOOP_POSITIONS_FT = ((5.25, 25.0), (88.75, 25.0))
COURT_LENGTH_FT = 94.0
ARC_3PT_FT = 23.75
CORNER_3PT_FT = 22.0
# The corner three line is straight for ~14 ft out from each baseline.
CORNER_ZONE_DEPTH_FT = 14.0


@dataclass
class ShotDetectorConfig:
    attempt_gap_s: float = 0.7     # max gap between shot-pose frames in one attempt
    make_window_s: float = 3.0     # seconds after release a make can register
    min_attempt_frames: int = 2    # shot pose must persist this many frames
    min_attempt_score: float = 0.5 # best shot-pose confidence in the window
    shooter_iou_min: float = 0.2   # min IoU between shot-pose box and a track


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def distance_to_nearest_hoop(x_ft: float, y_ft: float) -> Tuple[float, Tuple[float, float]]:
    """Distance (feet) from a court point to the nearest hoop, plus that hoop."""
    best = None
    best_hoop = HOOP_POSITIONS_FT[0]
    for hx, hy in HOOP_POSITIONS_FT:
        d = ((x_ft - hx) ** 2 + (y_ft - hy) ** 2) ** 0.5
        if best is None or d < best:
            best = d
            best_hoop = (hx, hy)
    return best, best_hoop


def classify_shot_type(x_ft: float, y_ft: float) -> Tuple[str, float]:
    """
    Classify a shot location as 2PT or 3PT and return (shot_type, distance_ft).

    Uses the NBA arc (23.75 ft) except in the corner zones near each baseline,
    where the line is straight at 22 ft.
    """
    dist, (hoop_x, _) = distance_to_nearest_hoop(x_ft, y_ft)
    baseline_x = 0.0 if hoop_x < COURT_LENGTH_FT / 2 else COURT_LENGTH_FT
    in_corner_zone = abs(x_ft - baseline_x) <= CORNER_ZONE_DEPTH_FT
    threshold = CORNER_3PT_FT if in_corner_zone else ARC_3PT_FT
    return ("3PT" if dist >= threshold else "2PT"), dist


def _group_frames(frame_indices: List[int], max_gap: int) -> List[Tuple[int, int]]:
    """Group sorted frame indices into (start, end) runs allowing gaps of max_gap."""
    if not frame_indices:
        return []
    frames = sorted(set(frame_indices))
    groups = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f - prev <= max_gap:
            prev = f
        else:
            groups.append((start, prev))
            start = prev = f
    groups.append((start, prev))
    return groups


def _best_detection_in(
    dets: List[Detection], frame_start: int, frame_end: int, class_ids: set
) -> Optional[Detection]:
    """Highest-confidence detection of the given classes inside a frame window."""
    best = None
    for d in dets:
        if d.class_id in class_ids and frame_start <= d.frame_idx <= frame_end:
            if best is None or d.score > best.score:
                best = d
    return best


def find_shooter_track(
    tracks: Any,
    shot_box: Sequence[float],
    frame_idx: int,
    iou_min: float,
    frame_tolerance: int = 5,
) -> Optional[int]:
    """
    Track id of the player whose box best overlaps the shot-pose box.

    Searches the release frame first, widening to +/- frame_tolerance frames
    so a short tracking dropout doesn't lose the attribution.
    """
    best_score = None
    best_track = None
    for t in tracks:
        if t.class_id not in PLAYER_CLASS_IDS:
            continue
        if abs(t.frame_idx - frame_idx) > frame_tolerance:
            continue
        overlap = iou(t.xyxy, shot_box)
        if overlap < iou_min:
            continue
        # Prefer closer frames on ties by penalizing frame distance slightly
        score = overlap - 0.001 * abs(t.frame_idx - frame_idx)
        if best_score is None or score > best_score:
            best_score = score
            best_track = t.track_id
    return best_track


def detect_shot_events(
    dets: List[Detection],
    tracks: Any,
    number_map: Dict[int, str],
    team_map: Dict[int, Any],
    segment_meta: dict,
    fps: float = 30.0,
    config: Optional[ShotDetectorConfig] = None,
) -> List[ShotEvent]:
    """
    Detect shot attempts and outcomes from frame-level detections.

    Returns ShotEvents with shooter/team attribution where possible and the
    shooter's image-space box stored for later homography projection
    (x_ft/y_ft/distance_ft/shot_type are finalized by the pipeline).
    """
    cfg = config or ShotDetectorConfig()
    gap_frames = max(1, int(round(cfg.attempt_gap_s * fps)))
    make_window_frames = max(1, int(round(cfg.make_window_s * fps)))

    pose_frames = [d.frame_idx for d in dets if d.class_id in SHOT_ATTEMPT_CLASS_IDS]
    attempt_windows = _group_frames(pose_frames, gap_frames)

    make_frames = [d.frame_idx for d in dets if d.class_id == MAKE_CLASS_ID]
    make_groups = _group_frames(make_frames, gap_frames)
    consumed_makes = set()

    period = int(segment_meta.get("period", 1))
    start_clock = float(segment_meta.get("start_clock_s", 0.0))
    end_clock = float(segment_meta.get("end_clock_s", 0.0))

    events: List[ShotEvent] = []
    for win_start, win_end in attempt_windows:
        # Require the pose to persist and to be confident at least once.
        n_pose_frames = len({
            d.frame_idx for d in dets
            if d.class_id in SHOT_ATTEMPT_CLASS_IDS and win_start <= d.frame_idx <= win_end
        })
        if n_pose_frames < cfg.min_attempt_frames:
            continue
        shot_det = _best_detection_in(dets, win_start, win_end, SHOT_ATTEMPT_CLASS_IDS)
        if shot_det is None or shot_det.score < cfg.min_attempt_score:
            continue

        release_frame = win_end

        # Make/miss: first unconsumed ball-in-basket group starting in the window
        # [attempt start, release + make_window].
        result = "miss"
        for mi, (m_start, _m_end) in enumerate(make_groups):
            if mi in consumed_makes:
                continue
            if win_start <= m_start <= release_frame + make_window_frames:
                result = "make"
                consumed_makes.add(mi)
                break

        # Shooter attribution
        shooter_track = find_shooter_track(
            tracks, shot_det.xyxy, shot_det.frame_idx, cfg.shooter_iou_min
        )
        team_id = team_map.get(shooter_track) if shooter_track is not None else None
        number = number_map.get(shooter_track) if shooter_track is not None else None

        # Game clock: linear interpolation across the segment
        clock = start_clock
        if fps > 0:
            clock = max(end_clock, start_clock - release_frame / fps)

        events.append(ShotEvent(
            period=period,
            game_clock_s=round(clock, 2),
            offense_team_id=str(team_id) if team_id is not None else None,
            defense_team_id=None,
            shooter_global_id=(
                f"{team_id}#{number}" if team_id is not None and number else None
            ),
            shooter_number=number or None,
            result=result,
            shot_type="2PT",  # refined by the pipeline once court coords exist
            x_ft=None,
            y_ft=None,
            distance_ft=None,
            video_frame_idx=release_frame,
            shooter_xyxy=tuple(shot_det.xyxy),
        ))

    return events
