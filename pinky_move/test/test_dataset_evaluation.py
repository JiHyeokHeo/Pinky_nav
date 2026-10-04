"""Regression checks for honest, non-mutating dataset label evaluation."""
import numpy as np
from evaluate_lane_dataset import ground_truth, masks_for


def test_polygon_label_is_scaled_without_changing_source(tmp_path):
    path = tmp_path/'label.txt'
    text = '1 0.1 0.2 0.2 0.2 0.2 0.8 0.1 0.8\n'
    path.write_text(text)
    items, errors = ground_truth(path, 640, 480)
    assert not errors and len(items) == 1
    np.testing.assert_allclose(items[0]['points'][0], [64, 96])
    assert len(masks_for(items, (480, 640))) == 1
    assert not masks_for(items, (480, 640), 0)
    assert path.read_text() == text


def test_bbox_label_is_not_presented_as_ground_truth_segmentation(tmp_path):
    path = tmp_path/'label.txt'
    path.write_text('1 0.5 0.5 0.2 0.2\n')
    items, errors = ground_truth(path, 640, 480)
    assert not items and 'bounding_box_not_segmentation' in errors[0]


def test_empty_label_and_missing_label_are_distinct(tmp_path):
    path = tmp_path/'label.txt'
    assert ground_truth(path, 640, 480)[1] == ['missing_label']
    path.write_text('')
    assert ground_truth(path, 640, 480) == ([], [])


def test_invalid_polygon_is_reported_not_silently_clamped(tmp_path):
    path = tmp_path/'label.txt'
    path.write_text('1 0 0 2 0 1 1\n')
    items, errors = ground_truth(path, 640, 480)
    assert not items and 'invalid_normalized_coordinates' in errors[0]
