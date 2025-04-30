"""Reward function for evaluating crossword puzzle answer formatting."""

import logging
import re
from typing import Any, List, Optional, Pattern

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class CrosswordFormatReward(RewardFunction):
    """
    Reward function for crossword puzzle game answers.

    Checks if completions follow the expected formatting for crossword puzzle answers:
    - Contains answer patterns like "1-Across: WORD"
    - Uses only valid characters (letters, no numbers or special chars in answers)
    - Follows specified formatting patterns
    """

    def __init__(
        self,
        format_patterns: Optional[List[Pattern]] = None,
        reward_value: float = 1.0,
        penalize_invalid_chars: bool = True,
        valid_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the crossword format reward function.

        Args:
            format_patterns: List of regex patterns to match (optional)
            reward_value: Value to award for correct formatting
            penalize_invalid_chars: Whether to penalize invalid characters
            valid_chars: String of valid characters for answers
            weight: Weight for this reward
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.reward_value = reward_value
        self.penalize_invalid_chars = penalize_invalid_chars
        self.valid_chars = valid_chars.upper()

        # Default patterns if none provided
        self.format_patterns = format_patterns or [
            re.compile(
                r"\d+-(?:Across|Down):\s+[A-Z\s]+", re.IGNORECASE
            ),  # Basic format pattern
            re.compile(
                r"^(?:\d+-(?:Across|Down):\s+[A-Z\s]+[\s,]*)+$", re.IGNORECASE
            ),  # Full response format
        ]

        # Precompile answer extraction regex for efficiency
        self._answer_regex = re.compile(
            r"(?:Across|Down):\s+([A-Za-z]+)", re.IGNORECASE
        )

        # Use set for valid char lookup for O(1) checks
        self._valid_chars_set = set(self.valid_chars)

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Check if completions follow crossword answer formatting.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context

        Returns:
            List of rewards (reward_value for correct format, 0.0 otherwise)
        """
        # Fast local variable/attribute resolution
        format_patterns = self.format_patterns
        penalize_invalid_chars = self.penalize_invalid_chars
        valid_chars_set = self._valid_chars_set
        answer_regex = self._answer_regex
        reward_value = self.reward_value

        rewards = []
        for completion in completions:
            try:
                content = self.get_content(completion)
                # Check for format patterns : returns True if any matches
                format_match = any(
                    pattern.search(content) for pattern in format_patterns
                )

                valid_chars = True
                if penalize_invalid_chars and format_match:
                    # Only check for valid chars if pattern matches
                    # Extract answers (text after "Across:" or "Down:")
                    for answer in answer_regex.findall(content):
                        # Check if answer contains only valid characters
                        for c in answer:
                            if c.upper() not in valid_chars_set:
                                valid_chars = False
                                break
                        if not valid_chars:
                            break

                # Both format and valid chars must be correct for full reward
                correct_format = format_match and valid_chars
                rewards.append(reward_value if correct_format else 0.0)

            except Exception as e:
                logger.error(f"Error in crossword format reward calculation: {e}")
                logger.exception(e)
                rewards.append(0.0)

        return rewards


# Legacy function for backward compatibility
def crossword_format_reward(completions: List[Any], **kwargs) -> List[float]:
    """
    Legacy function wrapper for CrosswordFormatReward.

    Args:
        completions: List of completions to evaluate
        **kwargs: Additional parameters

    Returns:
        List of rewards for crossword format quality
    """
    reward_fn = CrosswordFormatReward()
    return reward_fn.compute(completions, **kwargs)
