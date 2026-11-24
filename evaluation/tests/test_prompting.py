import pytest

from vlm_judge_bench.prompting import PredictionParseError, parse_prediction


def test_parse_fenced_json_and_clamp():
    text = '''```json
    {"no_error": true, "boxes": [{"bbox_1000_xyxy": {"x_min": -2, "y_min": 10, "x_max": 25, "y_max": 40}, "family": "D1_Artifact/A1_RenderingArtifact", "description": "smear"}]}
    ```'''
    no_error, boxes, warnings = parse_prediction(text)
    assert not no_error
    assert boxes[0].bbox_1000_xywh == (0.0, 10.0, 25.0, 30.0)
    assert len(warnings) == 2


def test_drop_invalid_box_and_unknown_family():
    text = '{"no_error": false, "boxes": [{"bbox_1000_xyxy": {"x_min":3,"y_min":2,"x_max":1,"y_max":4}}, {"bbox_1000_xyxy":{"x_min":1,"y_min":2,"x_max":4,"y_max":6}, "family":"x"}]}'
    no_error, boxes, warnings = parse_prediction(text)
    assert not no_error
    assert len(boxes) == 1 and boxes[0].family is None
    assert len(warnings) == 2


def test_missing_boxes_fails():
    with pytest.raises(PredictionParseError):
        parse_prediction('{"no_error": true}')


def test_parse_xyxy_list_to_canonical_xywh():
    _, boxes, warnings = parse_prediction('{"no_error":false,"boxes":[{"bbox_1000_xyxy":[10,20,40,80],"family":"D1_Artifact/A1_RenderingArtifact"}]}')
    assert boxes[0].bbox_1000_xywh == (10, 20, 30, 60)
    assert warnings == []
