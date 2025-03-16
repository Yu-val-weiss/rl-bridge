from collections import Counter

import numpy as np

from utils import mask_action_probs, random_argmax


def test_random_argmax_selection():
    np.random.seed(42)
    array = np.array([1, 3, 3, 2, 3])
    results = [random_argmax(array) for _ in range(1000)]
    expected_indices = {1, 2, 4}
    result_counts = Counter(results)
    assert set(result_counts.keys()).issubset(expected_indices)


def test_random_argmax_uniformity():
    np.random.seed(42)
    array = np.array([1, 3, 3, 2, 3])
    results = [random_argmax(array) for _ in range(1000)]
    result_counts = Counter(results)
    min_count = min(result_counts.values())
    max_count = max(result_counts.values())
    assert max_count - min_count < 100


def test_random_argmax_does_argmax():
    array = np.array([1, 19, 2, 2, 2, 2])
    assert random_argmax(array) == 1


def test_mask_action_probs_normal_case():
    action_probs = np.array([0.1, 0.2, 0.3, 0.4])
    legal_actions = [1, 3]
    masked_probs = mask_action_probs(action_probs, legal_actions)
    expected_probs = np.array([0.0, 0.33333333, 0.0, 0.66666667])
    np.testing.assert_allclose(masked_probs, expected_probs, rtol=1e-5)


def test_mask_action_probs_no_legal_actions():
    action_probs = np.array([0.1, 0.2, 0.3, 0.4])
    masked_probs_no_legal = mask_action_probs(action_probs, [])
    np.testing.assert_array_equal(masked_probs_no_legal, np.zeros_like(action_probs))


def test_mask_action_probs_all_legal():
    action_probs = np.array([0.1, 0.2, 0.3, 0.4])
    masked_probs_all_legal = mask_action_probs(action_probs, [0, 1, 2, 3])
    np.testing.assert_allclose(masked_probs_all_legal, action_probs, rtol=1e-5)


def test_mask_action_probs_single_legal_action():
    action_probs = np.array([0.1, 0.2, 0.3, 0.4])
    masked_probs_single = mask_action_probs(action_probs, [2])
    expected_single = np.array([0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(masked_probs_single, expected_single)
