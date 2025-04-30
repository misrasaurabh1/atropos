"""Reward function for evaluating step-by-step reasoning in completions."""

import logging
import re
from typing import Any, Dict, List, Optional

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class ReasoningStepsReward(RewardFunction):
    r"""
    Reward function that evaluates step-by-step reasoning in completions.

    Looks for several types of step-by-step reasoning indicators:
    1. Numbered step patterns like "Step 1:", "Step 2:"
    2. Numbered lists like "1.", "2." at start of line
    3. Bullet points with hyphens or asterisks
    4. Sequential transition words (First, Second, Next, Finally, etc.)
    """

    def __init__(
        self,
        min_words: int = 10,
        min_steps: int = 3,
        base_score: float = 0.1,
        pattern_weights: Optional[Dict[str, float]] = None,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the reasoning steps reward function.

        Args:
            min_words: Minimum number of words to consider for base score
            min_steps: Number of steps needed for full points in each category
            base_score: Base score for having content longer than min_words
            pattern_weights: Custom weights for each pattern type (optional)
            weight: Weight for this reward
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.min_words = min_words
        self.min_steps = min_steps
        self.base_score = base_score

        self.pattern_weights = {
            "numbered_steps": 0.5,
            "list_numbers": 0.5,
            "bullet_points": 0.4,
            "transition_words": 0.3,
        }
        if pattern_weights:
            self.pattern_weights.update(pattern_weights)

        # Pre-compile regex patterns for efficiency
        self.compiled_patterns = {
            "numbered_steps": re.compile(
                r"Step\s+\d+[\s:]+", re.IGNORECASE | re.MULTILINE
            ),
            "list_numbers": re.compile(
                r"(?:^|\n)\s*\d+\.\s+", re.IGNORECASE | re.MULTILINE
            ),
            "bullet_points": re.compile(
                r"(?:^|\n)\s*[\-\*•]\s+", re.IGNORECASE | re.MULTILINE
            ),
            "transition_words": re.compile(
                r"\b(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally|"
                r"Subsequently|Afterward|Lastly|Initially|To begin|Let\'s begin|"
                r"I\'ll first|After that|In conclusion|Eventually|Subsequently|"
                r"To solve|begin by|understand|analyze|apply|compute)\b",
                re.IGNORECASE | re.MULTILINE,
            ),
        }

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Calculate reasoning quality scores based on pattern matching.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context

        Returns:
            List of reward scores between 0.0 and 1.0
        """
        get_content = self.get_content
        min_steps = self.min_steps
        base_score = self.base_score
        min_words = self.min_words
        pattern_weights = self.pattern_weights
        compiled_patterns = self.compiled_patterns

        # Extract content outside the main scoring loop for speed
        completion_contents = [get_content(completion) for completion in completions]

        rewards = []
        for content in completion_contents:
            score = 0.0
            # Check matches for each pattern type
            for pattern_type, pattern in compiled_patterns.items():
                match_count = len(pattern.findall(content))
                weight = pattern_weights.get(pattern_type, 0.3)
                score += min(1.0, match_count / min_steps) * weight

            # Fast word count check for base score
            if self._fast_word_count(content) > min_words:
                score += base_score

            # Cap the total score at 1.0
            rewards.append(min(1.0, score))

        return rewards

    @staticmethod
    def _fast_word_count(text: str) -> int:
        # Count words using a simple fast method (splitting on whitespace-like chars, avoiding regex overhead)
        count = 0
        in_word = False
        for c in text:
            if c.isspace():
                in_word = False
            elif not in_word:
                count += 1
                in_word = True
        return count


# Legacy function for backward compatibility
def reasoning_steps_reward(completions: List[Any], **kwargs) -> List[float]:
    """
    Legacy function wrapper for ReasoningStepsReward.

    Args:
        completions: List of completions to evaluate
        **kwargs: Additional parameters

    Returns:
        List of reward scores between 0.0 and 1.0
    """
    reward_fn = ReasoningStepsReward()
    return reward_fn.compute(completions, **kwargs)
