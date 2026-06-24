"""Unit tests for temporal smoothing helpers."""

import numpy as np
import pytest

from hoopstats.smoothing import (
    PositionSmoother,
    apply_homography,
    build_dense_xy,
    dense_to_per_frame_positions,
    dense_to_trajectories,
    ema_homography,
    normalize_homography,
)


class TestNormalizeHomography:
    def test_scales_bottom_right_to_one(self):
        h = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)
        out = normalize_homography(h)
        assert out[2, 2] == pytest.approx(1.0)
        assert out[0, 0] == pytest.approx(1.0)

    def test_none_passes_through(self):
        assert normalize_homography(None) is None


class TestEmaHomography:
    def test_first_frame_returns_normalized_current(self):
        cur = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)
        out = ema_homography(None, cur, alpha=0.4)
        assert out[2, 2] == pytest.approx(1.0)

    def test_blends_prev_and_current(self):
        prev = np.eye(3)
        cur = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
        out = ema_homography(prev, cur, alpha=0.5)
        assert out[0, 0] == pytest.approx(1.5)

    def test_alpha_one_is_no_smoothing(self):
        prev = np.eye(3)
        cur = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=float)
        out = ema_homography(prev, cur, alpha=1.0)
        assert out[0, 0] == pytest.approx(3.0)


class TestApplyHomography:
    def test_identity(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = apply_homography(np.eye(3), pts)
        assert np.allclose(out, pts)

    def test_scaling(self):
        h = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=float)
        out = apply_homography(h, np.array([[1.0, 1.0]]))
        assert np.allclose(out, [[2.0, 3.0]])

    def test_perspective_divide(self):
        h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=float)
        out = apply_homography(h, np.array([[4.0, 6.0]]))
        assert np.allclose(out, [[2.0, 3.0]])  # divided by w=2

    def test_empty(self):
        out = apply_homography(np.eye(3), np.empty((0, 2)))
        assert out.shape == (0, 2)

    def test_matches_cv2(self):
        import cv2
        h = np.array([[1.2, 0.1, 5.0], [0.05, 1.1, -3.0], [0.0001, 0.0002, 1.0]])
        pts = np.array([[10.0, 20.0], [100.0, 200.0], [50.0, 75.0]])
        mine = apply_homography(h, pts)
        ref = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), h).reshape(-1, 2)
        assert np.allclose(mine, ref, atol=1e-6)


class TestDensePacking:
    def test_build_dense_xy_fills_present_and_nan(self):
        traj = {"1": [[10.0, 20.0, 0], [11.0, 21.0, 2]], "2": [[5.0, 5.0, 1]]}
        arr, present = build_dense_xy(traj, ["1", "2"], total_frames=3)
        assert arr.shape == (3, 2, 2)
        assert present.tolist() == [[True, False], [False, True], [True, False]]
        assert np.allclose(arr[0, 0], [10.0, 20.0])
        assert np.isnan(arr[1, 0]).all()  # track 1 absent at frame 1
        assert np.allclose(arr[1, 1], [5.0, 5.0])

    def test_build_dense_ignores_out_of_range_frames(self):
        traj = {"1": [[1.0, 1.0, 5]]}  # frame 5 beyond total_frames=3
        arr, present = build_dense_xy(traj, ["1"], total_frames=3)
        assert not present.any()

    def test_per_frame_positions_skips_absent(self):
        cleaned = np.array([[[10.0, 20.0], [0.0, 0.0]],
                            [[11.0, 21.0], [5.0, 5.0]]])
        present = np.array([[True, False], [True, True]])
        out = dense_to_per_frame_positions(cleaned, present, team_by_col=[0, 1])
        # frame 0: only column 0 present
        xy0, t0 = out[0]
        assert xy0.shape == (1, 2) and t0.tolist() == [0]
        # frame 1: both present
        xy1, t1 = out[1]
        assert xy1.shape == (2, 2) and sorted(t1.tolist()) == [0, 1]

    def test_dense_to_trajectories_roundtrip_present_only(self):
        cleaned = np.array([[[10.0, 20.0]], [[11.0, 21.0]], [[12.0, 22.0]]])
        present = np.array([[True], [False], [True]])
        out = dense_to_trajectories(cleaned, present, ["7"])
        assert out["7"] == [[10.0, 20.0, 0], [12.0, 22.0, 2]]

    def test_packing_survives_clean_paths(self):
        """End-to-end: pack -> clean_paths -> repack keeps a present-only path."""
        from sports import clean_paths
        traj = {"1": [[float(i), float(i), i] for i in range(12)]}  # smooth diagonal
        arr, present = build_dense_xy(traj, ["1"], total_frames=12)
        cleaned, _ = clean_paths(arr, smooth_window=5, smooth_poly=2)
        out = dense_to_trajectories(cleaned, present, ["1"])
        assert len(out["1"]) == 12  # every present frame retained


class TestPositionSmoother:
    def test_first_update_returns_input(self):
        s = PositionSmoother(alpha=0.5)
        assert np.allclose(s.update(1, [3.0, 4.0]), [3.0, 4.0])

    def test_ema_blends(self):
        s = PositionSmoother(alpha=0.5)
        s.update(1, [0.0, 0.0])
        assert np.allclose(s.update(1, [10.0, 10.0]), [5.0, 5.0])

    def test_keys_are_independent(self):
        s = PositionSmoother(alpha=0.5)
        s.update(1, [0.0, 0.0])
        s.update(2, [100.0, 100.0])
        assert np.allclose(s.update(1, [10.0, 10.0]), [5.0, 5.0])

    def test_prune_forgets_absent_keys(self):
        s = PositionSmoother(alpha=0.5)
        s.update(1, [0.0, 0.0])
        s.update(2, [0.0, 0.0])
        s.prune([1])
        # key 2 was pruned, so it restarts from the new value
        assert np.allclose(s.update(2, [10.0, 10.0]), [10.0, 10.0])

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            PositionSmoother(alpha=0.0)
        with pytest.raises(ValueError):
            PositionSmoother(alpha=1.5)
