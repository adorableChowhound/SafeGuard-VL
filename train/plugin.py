"""
GRPO Reward Plugin for SafeGuard-VL (Stage-2 RL Training)

Implements the verifiable reward function for policy-aware safe/unsafe
classification. Used with ms-swift's GRPO training pipeline.

Reward design:
  - Correct answer with valid <think> block: highest reward
  - Correct answer without thinking: moderate reward
  - Wrong answer with valid thinking: small reward
  - Invalid format: lowest reward
"""

import re
from typing import List

from swift.plugin import ORM, orms


class MultiModalAccuracyORM(ORM):

    def __init__(self, min_think_length=50):
        self.min_think_length = min_think_length
        self.valid_answers = {"yes", "no"}

    def _extract_answer(self, text: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_gt_answer(self, text: str) -> str:
        """Parse ground truth: <answer>safe</answer> -> 'yes', else -> 'no'."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
        if match and match.group(1).strip() == "safe":
            return "yes"
        return "no"

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []

        for content, sol in zip(completions, solution):
            # Check <think> block
            has_valid_think = False
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
            if think_match and len(think_match.group(1).strip()) >= self.min_think_length:
                has_valid_think = True

            # Parse ground truth
            gt_label = self._extract_gt_answer(sol)

            # Parse model prediction
            pred_ans = self._extract_answer(content)
            pred_is_valid = pred_ans.lower() in self.valid_answers

            # Compute reward
            if pred_is_valid:
                is_correct = pred_ans.lower() == gt_label
                if is_correct:
                    base = 1.7 if gt_label == "yes" else 2.0
                    reward = base if has_valid_think else base - 0.5
                else:
                    reward = 1.0 if has_valid_think else 0.5
            else:
                reward = 0.5 if has_valid_think else 0.0

            rewards.append(max(0.0, min(2.0, reward)))

        return rewards


# Register with ms-swift
orms["external_r1v_acc"] = MultiModalAccuracyORM
