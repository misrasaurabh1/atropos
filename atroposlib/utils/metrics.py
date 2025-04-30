import numpy as np


def get_std_min_max_avg(name: str, data: list, metrics_dict: dict) -> dict:
    """
    Calculate the standard deviation, minimum, maximum, and average of a list of numbers.
    Adds it to the wandb dict for logging.

    Args:
        data (list): A list of numbers.

    Returns:
        dict: A dictionary containing the standard deviation, minimum, maximum, and average.
    """
    arr = np.asarray(data)  # Convert to array once for better performance
    metrics_dict[f"{name}_mean"] = arr.mean()
    metrics_dict[f"{name}_std"] = arr.std()
    metrics_dict[f"{name}_max"] = arr.max()
    metrics_dict[f"{name}_min"] = arr.min()
    return metrics_dict
