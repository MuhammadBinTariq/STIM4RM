# mem_metrics/kenlm_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math

try:
    import kenlm
except ImportError as e:
    raise ImportError("kenlm is not installed. Run: pip install kenlm") from e


@dataclass
class KenLMScorer:
    """
    Thin wrapper to compute conditional log-prob under KenLM:
        log P(text + token | prefix) = log P(prefix + token) - log P(prefix)

    KenLM's .score returns log10 probability by default, so we convert to natural log.
    """
    model_path: str
    bos: bool = True
    eos: bool = False

    def __post_init__(self):
        self.model = kenlm.Model(self.model_path)

    @staticmethod
    def _log10_to_ln(x: float) -> float:
        return x * math.log(10.0)

    def score_sentence_ln(self, s: str) -> float:
        # KenLM score returns log10 P(sentence). Convert to ln.
        log10p = self.model.score(s, bos=self.bos, eos=self.eos)
        return self._log10_to_ln(log10p)

    def conditional_ln(self, prefix: str, next_piece: str) -> float:
        """
        Returns ln P(next_piece | prefix) approximated via sentence score difference.

        NOTE:
        - This treats prefix and next_piece as text concatenation.
        - Keep concatenation consistent with how you trained KenLM (word-level vs token-level).
        """
        prefix = prefix.strip()
        next_piece = next_piece.strip()

        if not prefix:
            return self.score_sentence_ln(next_piece)

        full = f"{prefix} {next_piece}"
        return self.score_sentence_ln(full) - self.score_sentence_ln(prefix)
