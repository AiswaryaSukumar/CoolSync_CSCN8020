# prompt_model/prompt_features.py

from dataclasses import dataclass, asdict
from typing import Dict, List


# Keep prompt labels centralized so every module uses the same names.
PROMPT_TYPES: List[str] = [
    "simple",
    "reasoning",
    "code",
    "long_context",
    "burst",
]


@dataclass
class PromptFeatures:
    """
    Structured representation of a prompt event.

    We intentionally avoid raw prompt text and instead represent
    each prompt using workload-relevant attributes.
    """

    prompt_type: str
    prompt_length: int
    complexity_score: float
    concurrency_level: int

    def to_dict(self) -> Dict:
        """Convert the dataclass into a dictionary for logging and serialization."""
        return asdict(self)