import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class RewardFunction(ABC):
    """Abstract base class for all reward functions"""

    def __init__(self, weight: float = 1.0, name: Optional[str] = None, **kwargs):
        """
        Initialize reward function with a weight and optional configuration.

        Args:
            weight: Importance factor when combining with other rewards
            name: Optional custom name for this reward function instance
            **kwargs: Additional configuration parameters specific to the reward function
        """
        self.weight = weight
        self._name = name
        self.config = kwargs
        self.wandb_logger = None

    @property
    def name(self) -> str:
        """Unique identifier for this reward function"""
        return self._name or self.__class__.__name__.lower()

    @abstractmethod
    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Compute reward scores for the given completions.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context like solution, ground_truth, etc.

        Returns:
            List of reward scores, one for each completion
        """
        pass

    def __call__(self, completions: List[Any], **kwargs) -> List[float]:
        """Wrapper that applies weight to the computed rewards"""
        try:
            rewards = self.compute(completions, **kwargs)
            # Apply weight
            weighted_rewards = [r * self.weight for r in rewards]

            # Log to wandb if available
            if self.wandb_logger:
                self.log_metrics(rewards, weighted_rewards)

            return weighted_rewards
        except Exception as e:
            logger.error(f"Error in reward function {self.name}: {e}")
            logger.exception(e)
            return [0.0] * len(completions)

    def set_wandb_logger(self, logger):
        """Set the WandB logger for this reward function"""
        self.wandb_logger = logger

    def log_metrics(self, raw_rewards: List[float], weighted_rewards: List[float]):
        """Log reward metrics to WandB"""
        # Move all checks to a single branch for early return
        wandb_logger = self.wandb_logger
        if wandb_logger is None or not raw_rewards:
            return

        name = self.name  # local var for speed & shorter strings
        rw_len = len(raw_rewards)
        wr_len = len(weighted_rewards)

        # Manual aggregation (only one traversal)
        rw_sum = 0.0
        rw_min = float("inf")
        rw_max = float("-inf")
        for r in raw_rewards:
            rw_sum += r
            if r < rw_min:
                rw_min = r
            if r > rw_max:
                rw_max = r

        wr_sum = 0.0
        for wr in weighted_rewards:
            wr_sum += wr

        # Format keys efficiently (avoid repeated string interpolation)
        prefix = f"reward/{name}/"
        metrics = {
            prefix + "mean_raw": rw_sum / rw_len,
            prefix + "mean_weighted": wr_sum / wr_len if wr_len else 0.0,
            prefix + "min": rw_min,
            prefix + "max": rw_max,
        }

        wandb_logger.log(metrics)

    @staticmethod
    def get_content(completion: Any) -> str:
        """
        Extract content from different completion formats.

        Supports:
          - String completions
          - Dict with {"role": "assistant", "content": "text"}
          - Dict with {"message": {"role": "assistant", "content": "text"}}
          - List of messages where one has role "assistant"

        Args:
            completion: The completion in any supported format

        Returns:
            The extracted content as a string
        """
        if isinstance(completion, str):
            return completion
        elif isinstance(completion, dict):
            if (
                "role" in completion
                and completion["role"] == "assistant"
                and "content" in completion
            ):
                return completion["content"]
            if "message" in completion and isinstance(completion["message"], dict):
                if (
                    "role" in completion["message"]
                    and completion["message"]["role"] == "assistant"
                    and "content" in completion["message"]
                ):
                    return completion["message"]["content"]
        elif isinstance(completion, list) and len(completion) > 0:
            # Look for assistant messages
            for msg in completion:
                if (
                    isinstance(msg, dict)
                    and "role" in msg
                    and msg["role"] == "assistant"
                    and "content" in msg
                ):
                    return msg["content"]

        # If no assistant content found, return empty string
        return ""

    @property
    def name(self) -> str:
        # Avoid recomputation; cached property pattern.
        n = self._name
        if n is not None:
            return n
        # Could fallback to class name or other logic if needed
        return self.__class__.__name__
