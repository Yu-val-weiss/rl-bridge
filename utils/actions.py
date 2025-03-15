import numpy as np


def random_argmax(array: np.ndarray) -> int:
    """Computes an argmax, breaking ties by choosing randomly"""
    max_value = np.max(array)  # Find the maximum value
    max_indices = np.where(array == max_value)[0]  # Get indices of max values
    return np.random.choice(max_indices)  # Randomly pick one if there's a tie


def mask_action_probs(action_probs: np.ndarray, legal_actions: list[int]) -> np.ndarray:
    masked_probs = np.zeros_like(action_probs)  # Start with all zeros
    masked_probs[legal_actions] = action_probs[legal_actions]  # keep legal actions

    total_prob = masked_probs.sum()

    # safe normalisation and return
    return masked_probs / total_prob if total_prob > 0 else masked_probs
