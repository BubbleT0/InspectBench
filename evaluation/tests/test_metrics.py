import math

import pytest

from vlm_judge_bench.data import GTBox, PredictionBox, PredictionRecord, Sample
from vlm_judge_bench.metrics import box_pair_stats, compute_metrics, evaluate_sample, paired_bootstrap


FAMILY = "D1_Artifact/A2_StructuralArtifact"
TEXT_FAMILY = "D2_TextCorrectness/A2_TextContentFidelity"


def sample(sample_id, boxes):
    return Sample(sample_id, "/unused.png", 1000, 1000, "", not boxes, tuple(GTBox(tuple(b), FAMILY) for b in boxes), "test", 1)


def sourced_sample(sample_id, boxes, source):
    value = sample(sample_id, boxes)
    return Sample(
        value.sample_id, value.image_path, value.image_width, value.image_height,
        value.prompt, value.no_error, value.boxes, source, value.source_line,
    )


def pred(sample_id, boxes):
    return PredictionRecord(sample_id, "m", "ok", [PredictionBox(tuple(b), FAMILY) for b in boxes], not boxes)


def categorized_sample(sample_id, boxes):
    return Sample(sample_id, "/unused.png", 1000, 1000, "", False, tuple(
        GTBox(tuple(box), family) for box, family in boxes
    ), "test", 1)


def test_pair_score_perfect_and_no_overlap():
    assert box_pair_stats((10, 10, 20, 20), (10, 10, 20, 20), 1_000_000).score == pytest.approx(1)
    assert box_pair_stats((100, 100, 20, 20), (10, 10, 20, 20), 1_000_000).score == 0


def test_one_prediction_can_claim_multiple_ground_truth_boxes():
    s = sample("s", [(10, 10, 10, 10), (30, 10, 10, 10)])
    row = evaluate_sample(s, pred("s", [(10, 10, 30, 10)]))
    assert len(row["matched"]) == 2
    assert {pred_idx for pred_idx, _, _ in row["matched"]} == {0}
    assert row["soft_recall"] > 0.5
    assert row["image_score"] > 0.5
    assert row["merge_predictions"] == 1


def test_clean_penalty_and_headline():
    positive = sample("p", [(10, 10, 20, 20)])
    clean = sample("c", [])
    predictions = {"p": pred("p", [(10, 10, 20, 20)]), "c": pred("c", [(50, 50, 20, 20)])}
    metrics, rows = compute_metrics([positive, clean], predictions)
    assert rows[0]["image_score"] == pytest.approx(1)
    assert rows[1]["image_score"] == pytest.approx(math.exp(-1))
    expected = 100 * (0.7 + 0.3 * math.sqrt(math.exp(-1))) ** 2
    assert metrics["headline"]["elastic_bbox_score"] == pytest.approx(expected)
    assert metrics["headline"]["clean_fppi"] == 1


def test_region_group_scores_follow_gt_categories_and_localization_quality():
    mixed = categorized_sample("mixed", [
        ((10, 10, 20, 20), FAMILY),
        ((100, 100, 20, 20), TEXT_FAMILY),
    ])
    # Prediction family is deliberately Artifact for both boxes: group membership
    # must come from the human GT family, not the model-provided label.
    predictions = {"mixed": pred("mixed", [(10, 10, 20, 20)])}
    metrics, rows = compute_metrics([mixed], predictions)
    assert rows[0]["gt_families"] == [FAMILY, TEXT_FAMILY]
    assert metrics["headline"]["artifact_region_bbox_score"] == pytest.approx(100)
    assert metrics["headline"]["text_region_bbox_score"] == pytest.approx(0)
    assert metrics["region_group_scores"]["artifact"]["gt_boxes"] == 1
    assert metrics["region_group_scores"]["text_correctness"]["gt_boxes"] == 1


def test_image_level_binary_decision_error_rates():
    positive_hit = sample("positive_hit", [(10, 10, 20, 20)])
    positive_miss = sample("positive_miss", [(40, 40, 20, 20)])
    clean_ok = sample("clean_ok", [])
    clean_false_alarm = sample("clean_false_alarm", [])
    predictions = {
        "positive_hit": pred("positive_hit", [(10, 10, 20, 20)]),
        "positive_miss": pred("positive_miss", []),
        "clean_ok": pred("clean_ok", []),
        "clean_false_alarm": pred("clean_false_alarm", [(100, 100, 20, 20)]),
    }
    metrics, _ = compute_metrics(
        [positive_hit, positive_miss, clean_ok, clean_false_alarm],
        predictions,
    )
    binary = metrics["binary_decision"]
    assert binary["positive_false_negative_rate"] == pytest.approx(0.5)
    assert binary["clean_false_positive_rate"] == pytest.approx(0.5)
    assert binary["overall_error_rate"] == pytest.approx(0.5)
    assert binary["positive_false_negatives"] == 1
    assert binary["clean_false_positives"] == 1
    assert binary["total_errors"] == 2


def test_duplicate_prediction_and_scale_metrics():
    s = sample("s", [(10, 10, 20, 20)])
    predictions = {"s": pred("s", [(10, 10, 20, 20), (10, 10, 20, 20)])}
    metrics, _ = compute_metrics([s], predictions)
    assert metrics["multi_box"]["duplicate_fp"] == 1
    assert metrics["scale"]["area_ratio"]["median"] == pytest.approx(1)


def test_duplicate_and_background_predictions_receive_zero_credit():
    s = sample("s", [(10, 10, 20, 20)])
    row = evaluate_sample(s, pred("s", [
        (10, 10, 20, 20),
        (10, 10, 20, 20),
        (500, 500, 20, 20),
    ]))
    assert row["prediction_credits"] == pytest.approx([1, 0, 0])
    assert row["soft_recall"] == pytest.approx(1)
    assert row["soft_precision"] == pytest.approx(1 / 3)
    assert row["image_score"] == pytest.approx(0.5)
    assert row["duplicate_fp"] == 1
    assert row["background_fp"] == 1


def test_contiguous_merged_box_can_score_perfectly():
    s = sample("s", [(10, 10, 10, 10), (20, 10, 10, 10), (30, 10, 10, 10)])
    row = evaluate_sample(s, pred("s", [(10, 10, 30, 10)]))
    assert row["gt_scores"] == pytest.approx([1, 1, 1])
    assert row["soft_precision"] == pytest.approx(1)
    assert row["soft_recall"] == pytest.approx(1)
    assert row["image_score"] == pytest.approx(1)


def test_paired_bootstrap_perfect_model_wins():
    positive = sample("p", [(10, 10, 20, 20)])
    clean = sample("c", [])
    perfect = {"p": pred("p", [(10, 10, 20, 20)]), "c": pred("c", [])}
    empty = {"p": pred("p", []), "c": pred("c", [])}
    _, perfect_rows = compute_metrics([positive, clean], perfect)
    _, empty_rows = compute_metrics([positive, clean], empty)
    comparison = paired_bootstrap({"perfect": perfect_rows, "empty": empty_rows}, iterations=20, seed=1)
    delta = comparison["pairs"]["empty_minus_perfect"]["elastic_bbox_score"]
    assert delta["mean_delta"] < 0
    assert delta["probability_gt_0"] == 0


def test_annotation_source_buckets():
    samples = [
        sourced_sample("g", [(10, 10, 20, 20)], "/x/GPT_completed_211_x.jsonl"),
        sourced_sample("u", [], "/x/U1.5_completed_130_x.jsonl"),
        sourced_sample("e", [], "/x/External4_completed_210_x.jsonl"),
    ]
    predictions = {item.sample_id: pred(item.sample_id, []) for item in samples}
    metrics, _ = compute_metrics(samples, predictions)
    assert list(metrics["buckets_by_source"]) == ["gpt", "u1_5", "external4"]
    assert metrics["buckets_by_source"]["gpt"]["positive_images"] == 1
    assert metrics["buckets_by_source"]["u1_5"]["clean_images"] == 1
