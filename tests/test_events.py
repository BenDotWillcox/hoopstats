"""Unit tests for the shot event detector (synthetic detections, no models)."""

import pytest

from hoopstats.detection import Detection
from hoopstats.events import (
    ShotDetectorConfig,
    classify_shot_type,
    detect_shot_events,
    distance_to_nearest_hoop,
    find_shooter_track,
    iou,
)
from hoopstats.tracking import TrackedObject

FPS = 30.0
META = {"period": 1, "start_clock_s": 600.0, "end_clock_s": 0.0}


def det(frame_idx, class_id, xyxy=(100, 100, 150, 200), score=0.9):
    return Detection(frame_idx=frame_idx, cls=str(class_id), class_id=class_id,
                     xyxy=xyxy, score=score)


def track(frame_idx, track_id, xyxy=(100, 100, 150, 200), class_id=3):
    return TrackedObject(frame_idx=frame_idx, track_id=track_id, cls=str(class_id),
                         class_id=class_id, xyxy=xyxy, score=0.9)


def jump_shot_window(start, length=5, **kwargs):
    return [det(start + i, 5, **kwargs) for i in range(length)]


class TestAttemptGrouping:
    def test_single_attempt_detected(self):
        dets = jump_shot_window(10)
        evts = detect_shot_events(dets, [], {}, {}, META, fps=FPS)
        assert len(evts) == 1
        assert evts[0].result == "miss"
        assert evts[0].video_frame_idx == 14  # release = last pose frame

    def test_pose_frames_with_small_gap_are_one_attempt(self):
        dets = jump_shot_window(10, 3) + jump_shot_window(20, 3)  # 7-frame gap < 0.7s @30fps
        evts = detect_shot_events(dets, [], {}, {}, META, fps=FPS)
        assert len(evts) == 1

    def test_distant_windows_are_separate_attempts(self):
        dets = jump_shot_window(10) + jump_shot_window(200)
        evts = detect_shot_events(dets, [], {}, {}, META, fps=FPS)
        assert len(evts) == 2

    def test_single_frame_pose_is_filtered(self):
        dets = [det(10, 5)]
        evts = detect_shot_events(dets, [], {}, {}, META, fps=FPS)
        assert evts == []

    def test_low_confidence_attempt_is_filtered(self):
        dets = jump_shot_window(10, score=0.3)
        evts = detect_shot_events(dets, [], {}, {}, META, fps=FPS)
        assert evts == []

    def test_layup_dunk_class_also_counts(self):
        dets = [det(10 + i, 6) for i in range(4)]
        evts = detect_shot_events(dets, [], {}, {}, META, fps=FPS)
        assert len(evts) == 1


class TestMakeMiss:
    def test_make_within_window(self):
        dets = jump_shot_window(10) + [det(50, 1), det(51, 1)]
        evts = detect_shot_events(dets, [], {}, {}, META, fps=FPS)
        assert len(evts) == 1
        assert evts[0].result == "make"

    def test_ball_in_basket_too_late_is_miss(self):
        # release at frame 14, make window = 3s @ 30fps = 90 frames -> 200 is too late
        dets = jump_shot_window(10) + [det(200, 1)]
        evts = detect_shot_events(dets, [], {}, {}, META, fps=FPS)
        assert evts[0].result == "miss"

    def test_make_before_attempt_not_consumed(self):
        dets = [det(2, 1)] + jump_shot_window(10)
        evts = detect_shot_events(dets, [], {}, {}, META, fps=FPS)
        assert evts[0].result == "miss"

    def test_one_make_only_credits_one_attempt(self):
        # Two attempts, one ball-in-basket after the second.
        dets = (jump_shot_window(10) + jump_shot_window(150)
                + [det(180, 1)])
        evts = detect_shot_events(dets, [], {}, {}, META, fps=FPS)
        assert len(evts) == 2
        results = [e.result for e in sorted(evts, key=lambda e: e.video_frame_idx)]
        assert results == ["miss", "make"]


class TestShooterAttribution:
    def test_shooter_track_team_and_number(self):
        shot_box = (100, 100, 150, 200)
        dets = jump_shot_window(10, xyxy=shot_box)
        tracks = [
            track(14, track_id=7, xyxy=(102, 98, 152, 198)),   # overlaps shot box
            track(14, track_id=9, xyxy=(400, 100, 450, 200)),  # far away
        ]
        evts = detect_shot_events(
            dets, tracks, number_map={7: "23"}, team_map={7: 0, 9: 1},
            segment_meta=META, fps=FPS,
        )
        assert evts[0].offense_team_id == "0"
        assert evts[0].shooter_number == "23"
        assert evts[0].shooter_global_id == "0#23"
        assert evts[0].shooter_xyxy == shot_box

    def test_no_overlapping_track_leaves_shooter_unknown(self):
        dets = jump_shot_window(10)
        tracks = [track(14, track_id=9, xyxy=(400, 100, 450, 200))]
        evts = detect_shot_events(dets, tracks, {}, {9: 1}, META, fps=FPS)
        assert evts[0].offense_team_id is None
        assert evts[0].shooter_number is None

    def test_find_shooter_prefers_higher_iou(self):
        shot_box = (100, 100, 150, 200)
        tracks = [
            track(14, track_id=1, xyxy=(120, 100, 170, 200)),  # partial overlap
            track(14, track_id=2, xyxy=(101, 101, 151, 201)),  # near-perfect
        ]
        assert find_shooter_track(tracks, shot_box, 14, iou_min=0.2) == 2

    def test_find_shooter_widens_to_nearby_frames(self):
        shot_box = (100, 100, 150, 200)
        tracks = [track(17, track_id=4, xyxy=shot_box)]  # 3 frames after release
        assert find_shooter_track(tracks, shot_box, 14, iou_min=0.2) == 4

    def test_referee_track_is_ignored(self):
        shot_box = (100, 100, 150, 200)
        tracks = [track(14, track_id=4, xyxy=shot_box, class_id=8)]  # referee
        assert find_shooter_track(tracks, shot_box, 14, iou_min=0.2) is None


class TestGameClock:
    def test_clock_interpolates_from_segment_meta(self):
        dets = jump_shot_window(10)  # release at frame 14
        evts = detect_shot_events(dets, [], {}, {}, META, fps=FPS)
        assert evts[0].period == 1
        assert evts[0].game_clock_s == pytest.approx(600.0 - 14 / 30.0, abs=0.01)


class TestCourtGeometry:
    def test_distance_uses_nearest_hoop(self):
        d_left, hoop = distance_to_nearest_hoop(10.0, 25.0)
        assert hoop == (5.25, 25.0)
        assert d_left == pytest.approx(4.75)
        d_right, hoop = distance_to_nearest_hoop(85.0, 25.0)
        assert hoop == (88.75, 25.0)

    def test_layup_is_2pt(self):
        shot_type, dist = classify_shot_type(7.0, 25.0)
        assert shot_type == "2PT"
        assert dist < 3

    def test_top_of_key_three(self):
        shot_type, dist = classify_shot_type(30.0, 25.0)
        assert shot_type == "3PT"
        assert dist == pytest.approx(24.75)

    def test_corner_three_uses_22ft_line(self):
        # In the corner zone (x <= 14), 22-23.75 ft is still a three.
        shot_type, dist = classify_shot_type(2.0, 2.0)
        assert 22.0 <= dist < 23.75
        assert shot_type == "3PT"

    def test_midrange_above_break_is_2pt(self):
        # 23 ft would be a corner three, but above the break it's inside the arc.
        shot_type, dist = classify_shot_type(28.0, 25.0)
        assert shot_type == "2PT"
        assert dist == pytest.approx(22.75)

    def test_mirrored_on_right_half(self):
        shot_type, _ = classify_shot_type(92.0, 48.0)
        assert shot_type == "3PT"


class TestIoU:
    def test_identical_boxes(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_half_overlap(self):
        assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_custom_config_thresholds():
    cfg = ShotDetectorConfig(min_attempt_frames=1, min_attempt_score=0.1)
    dets = [det(10, 5, score=0.2)]
    evts = detect_shot_events(dets, [], {}, {}, META, fps=FPS, config=cfg)
    assert len(evts) == 1
