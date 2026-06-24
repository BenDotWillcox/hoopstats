"""Unit tests for evaluation matching/metrics and annotation CSV round-trip."""

import pytest

from hoopstats.annotate import ShotLabel, load_labels, save_labels
from hoopstats.evaluate import compute_metrics, format_report, match_events

FPS = 30.0


def pred(frame, result="make", team="0", number="23", shot_type="2PT"):
    return {
        "video_frame_idx": str(frame),
        "result": result,
        "offense_team_id": team,
        "shooter_number": number,
        "shot_type": shot_type,
    }


def label(frame, result="make", team="0", number="23", zone="paint"):
    return {
        "video_frame_idx": str(frame),
        "time_s": str(frame / FPS),
        "result": result,
        "offense_team_id": team,
        "shooter_number": number,
        "zone": zone,
    }


class TestMatchEvents:
    def test_exact_match(self):
        assert match_events([100], [100], 60) == [(0, 0)]

    def test_within_tolerance(self):
        assert match_events([100], [150], 60) == [(0, 0)]

    def test_outside_tolerance_no_match(self):
        assert match_events([100], [200], 60) == []

    def test_each_event_matches_once(self):
        # Two predictions near one label: only the closer one matches.
        pairs = match_events([95, 110], [100], 60)
        assert pairs == [(0, 0)]

    def test_closest_pairs_win(self):
        # pred 0 could match either label; greedy assigns by distance.
        pairs = match_events([100, 130], [128, 102], 60)
        assert sorted(pairs) == [(0, 1), (1, 0)]

    def test_empty_inputs(self):
        assert match_events([], [100], 60) == []
        assert match_events([100], [], 60) == []


class TestComputeMetrics:
    def test_perfect_prediction(self):
        # Two teams so the permutation-invariant team metric is meaningful.
        preds = [pred(100, team="0"), pred(300, result="miss", team="1")]
        labels = [label(102, team="0"), label(295, result="miss", team="1")]
        m = compute_metrics(preds, labels, fps=FPS)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["result_accuracy"] == 1.0
        assert m["team_accuracy"] == 1.0
        assert m["number_accuracy"] == 1.0
        assert m["shot_type_accuracy"] == 1.0
        assert m["false_positive_frames"] == []
        assert m["missed_label_frames"] == []

    def test_false_positive_and_miss(self):
        preds = [pred(100), pred(500)]      # 500 is a false positive
        labels = [label(102), label(900)]   # 900 was missed
        m = compute_metrics(preds, labels, fps=FPS)
        assert m["n_matched"] == 1
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5
        assert m["false_positive_frames"] == [500]
        assert m["missed_label_frames"] == [900]

    def test_make_miss_disagreement(self):
        preds = [pred(100, result="miss")]
        labels = [label(100, result="make")]
        m = compute_metrics(preds, labels, fps=FPS)
        assert m["result_accuracy"] == 0.0

    def test_blank_labels_excluded_from_attribution(self):
        # Annotator couldn't tell team/number/zone -> not counted against us.
        preds = [pred(100, team="1", number="99")]
        labels = [label(100, team="", number="", zone="")]
        m = compute_metrics(preds, labels, fps=FPS)
        assert m["team_accuracy"] is None
        assert m["number_accuracy"] is None
        assert m["shot_type_accuracy"] is None
        assert m["result_n"] == 1  # make/miss always compared

    def test_zone_to_shot_type(self):
        preds = [pred(100, shot_type="3PT"), pred(300, shot_type="2PT")]
        labels = [label(100, zone="corner3"), label(300, zone="arc3")]
        m = compute_metrics(preds, labels, fps=FPS)
        assert m["shot_type_accuracy"] == 0.5  # 3PT/corner3 right, 2PT/arc3 wrong

    def test_tolerance_scales_with_fps(self):
        preds = [pred(100)]
        labels = [label(200)]
        assert compute_metrics(preds, labels, fps=30.0)["n_matched"] == 0
        assert compute_metrics(preds, labels, fps=60.0)["n_matched"] == 1

    def test_empty_predictions(self):
        m = compute_metrics([], [label(100)], fps=FPS)
        assert m["precision"] is None
        assert m["recall"] == 0.0
        assert m["missed_label_frames"] == [100]


class TestFormatReport:
    def test_report_contains_key_metrics(self):
        m = compute_metrics([pred(100)], [label(100)], fps=FPS)
        report = format_report(m)
        assert "Precision: 100.0%" in report
        assert "Make/miss:  100.0% (1/1)" in report
        assert "| 100 | 100 |" in report

    def test_report_handles_no_matches(self):
        m = compute_metrics([], [], fps=FPS)
        report = format_report(m)
        assert "n/a" in report


class TestLabelRoundTrip:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "gt.csv"
        labels = [
            ShotLabel(video_frame_idx=300, time_s=10.0, result="miss",
                      offense_team_id="1", shooter_number="7", zone="arc3"),
            ShotLabel(video_frame_idx=100, time_s=3.33, result="make"),
        ]
        save_labels(labels, path)
        loaded = load_labels(path)
        # Sorted by frame on save
        assert [l.video_frame_idx for l in loaded] == [100, 300]
        assert loaded[0].result == "make"
        assert loaded[0].offense_team_id == ""
        assert loaded[1].zone == "arc3"

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_labels(tmp_path / "nope.csv") == []

    def test_labels_feed_evaluation(self, tmp_path):
        """Annotator output is directly consumable by compute_metrics."""
        import csv
        path = tmp_path / "gt.csv"
        save_labels([ShotLabel(video_frame_idx=100, time_s=3.33, result="make",
                               offense_team_id="0", shooter_number="23",
                               zone="paint")], path)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        m = compute_metrics([pred(100)], rows, fps=FPS)
        assert m["n_matched"] == 1
        assert m["result_accuracy"] == 1.0
