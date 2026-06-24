"""
Temporal smoothing for court-mapped tracking.

The court-map video computes a homography per frame and projects each player's
foot point independently, so raw top-down positions jitter from both detection
noise and frame-to-frame homography wobble. These helpers smooth two things:

1. The homography matrix across frames (EMA on the normalized 3x3), which also
   lets us reuse the last good homography on frames where keypoints fail
   (gap-fill) instead of dropping players entirely.
2. Each tracked player's (and the ball's) projected court position (per-track
   EMA).

All functions are pure NumPy (no cv2) so they can be unit tested without video.
"""

from typing import Dict, Optional

import numpy as np


def normalize_homography(h: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Scale a 3x3 homography so the bottom-right entry is 1 (removes the
    arbitrary scale factor, making matrices comparable for averaging)."""
    if h is None:
        return None
    h = np.asarray(h, dtype=float)
    if abs(h[2, 2]) > 1e-12:
        return h / h[2, 2]
    return h


def ema_homography(
    prev: Optional[np.ndarray], cur: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Exponential moving average of homography matrices.

    alpha is the weight on the current frame (1.0 = no smoothing). Both inputs
    are normalized first so the EMA is over comparable scales.
    """
    cur_n = normalize_homography(cur)
    if prev is None:
        return cur_n
    prev_n = normalize_homography(prev)
    return normalize_homography(alpha * cur_n + (1.0 - alpha) * prev_n)


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Project (N, 2) image points to court coordinates with a 3x3 homography.

    Pure-NumPy equivalent of cv2.perspectiveTransform.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 2)
    pts = pts.reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))])  # (N, 3)
    proj = hom @ np.asarray(h, dtype=float).T       # (N, 3)
    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return proj[:, :2] / w


def build_dense_xy(trajectories, columns, total_frames):
    """
    Pack sparse per-track trajectories into the dense (T, P, 2) array that
    `sports.clean_paths` expects, plus a (T, P) presence mask.

    trajectories: {key: [[x, y, frame], ...]}; columns: ordered keys -> P axis.
    Frames where a track is absent are NaN (and present=False), so cleaning can
    smooth/interpolate within a track without inventing it where it never was.
    """
    P = len(columns)
    arr = np.full((total_frames, P, 2), np.nan, dtype=float)
    present = np.zeros((total_frames, P), dtype=bool)
    for col, key in enumerate(columns):
        for x, y, frame in trajectories[key]:
            f = int(frame)
            if 0 <= f < total_frames:
                arr[f, col] = (x, y)
                present[f, col] = True
    return arr, present


def dense_to_per_frame_positions(cleaned, present, team_by_col):
    """
    Repack a cleaned (T, P, 2) array into {frame: (xy (N,2), teams (N,))} for
    rendering. Only frames where a track was actually present are emitted, so
    long absences are never rendered as phantom glides across the court.
    """
    T, P, _ = cleaned.shape
    out = {}
    for t in range(T):
        xs, teams = [], []
        for col in range(P):
            if present[t, col] and np.all(np.isfinite(cleaned[t, col])):
                xs.append(cleaned[t, col])
                teams.append(team_by_col[col])
        out[t] = (np.array(xs) if xs else np.empty((0, 2)), np.array(teams))
    return out


def dense_to_trajectories(cleaned, present, columns):
    """Repack a cleaned (T, P, 2) array back to {key: [[x, y, frame], ...]},
    keeping only frames where the track was present."""
    T, P, _ = cleaned.shape
    out = {}
    for col, key in enumerate(columns):
        pts = []
        for t in range(T):
            if present[t, col] and np.all(np.isfinite(cleaned[t, col])):
                pts.append([float(cleaned[t, col, 0]), float(cleaned[t, col, 1]), int(t)])
        out[key] = pts
    return out


class PositionSmoother:
    """Per-key exponential moving average of 2-D positions.

    Keyed by tracker id (and a sentinel for the ball), so each entity is
    smoothed along its own trajectory. New keys start un-smoothed; missing keys
    can be pruned so a returning player doesn't snap from a stale position.
    """

    def __init__(self, alpha: float = 0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.state: Dict[object, np.ndarray] = {}

    def update(self, key, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        prev = self.state.get(key)
        smoothed = xy if prev is None else self.alpha * xy + (1.0 - self.alpha) * prev
        self.state[key] = smoothed
        return smoothed

    def prune(self, live_keys) -> None:
        """Forget any keys not in live_keys (so re-appearances restart clean)."""
        live = set(live_keys)
        for key in list(self.state):
            if key not in live:
                del self.state[key]
