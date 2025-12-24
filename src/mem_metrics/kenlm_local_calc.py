# mem_metrics/kenlm_local_calc.py
from __future__ import annotations
from typing import List, Dict
from scipy.stats import spearmanr

from mem_metrics.mem import MemCalculator
from mem_metrics.kenlm_utils import KenLMScorer


class KenLMLocalCalculator(MemCalculator):
    """
    Drop-in replacement for the Infini-gram LocalCalculator:
    - Instead of ele['fre'] = infinigram_count(...)
    - We set ele['fre'] = KenLM conditional logprob (ln) for compatibility
    """

    def __init__(
        self,
        model_input,
        model_output,
        is_correct,
        model,
        tokenizer_olmo,
        task_type,
        original_str=None,
        step=None,
        k=20,
        kenlm_path: str = "dolma.bin",
        # Optional: limit prefix length if you want to approximate n-gram context
        max_prefix_chars: int = 2000,
        index: str = "unused_now",  # kept to preserve your signature if other code passes it
    ):
        super().__init__(
            model_input=model_input,
            model_output=model_output,
            is_correct=is_correct,
            model=model,
            tokenizer_olmo=tokenizer_olmo,
            k=k,
            task_type=task_type,
            original_str=original_str,
            index=index,
            step=step,
        )
        self.kenlm = KenLMScorer(kenlm_path, bos=True, eos=False)
        self.max_prefix_chars = max_prefix_chars

    def _make_prefix_text(self, start: int) -> str:
        """
        Produce the prefix string used for KenLM conditioning.
        This mirrors your old logic: model_input + model_output[:start]
        """
        prefix = (self.model_input + self.model_output[:start]).strip()
        if self.max_prefix_chars and len(prefix) > self.max_prefix_chars:
            prefix = prefix[-self.max_prefix_chars :]
        return prefix

    def get_fre(self, token_alternative_ls: List[Dict]) -> List[Dict]:
        """
        Output format is kept identical to your Infini-gram version.
        We store KenLM conditional ln-prob in the field named `fre` so downstream code works.
        """
        if not token_alternative_ls:
            return []

        out = []
        for ele in token_alternative_ls:
            start = ele["start"]
            prefix = self._make_prefix_text(start)

            # Main token (for compatibility we keep 'longest_nonzero_seq' but it’s no longer used)
            tok = ele["token"]
            ele["longest_nonzero_seq"] = ""  # not meaningful with KenLM
            ele["fre"] = float(self.kenlm.conditional_ln(prefix, tok))  # ln prob, not frequency

            # Alternatives: each at has al_token + prob
            alternative_tokens = ele.get("alternative_tokens", [])
            for j, at in enumerate(alternative_tokens):
                al_tok = at["al_token"]
                at["al_seq"] = ""  # keep key, but not needed
                at["fre"] = float(self.kenlm.conditional_ln(prefix, al_tok))
                alternative_tokens[j] = at

            ele["alternative_tokens"] = alternative_tokens
            out.append(ele)

        return out

    def cal_score(self, token_alternative_fre_ls: List[Dict]) -> List[Dict]:
        """
        Same as your existing cal_score(), except that `fre` is now KenLM ln-prob.
        Spearman is rank-based, so absolute scale doesn’t matter.
        """
        if not token_alternative_fre_ls:
            return []

        for i, ele in enumerate(token_alternative_fre_ls):
            fre_ls, prob_ls = [], []
            alts = ele.get("alternative_tokens", [])

            # top-k by model probability
            alts = sorted(alts, key=lambda x: x["prob"], reverse=True)[: self.k]

            for at in alts:
                fre_ls.append(at["fre"])   # KenLM conditional ln-prob
                prob_ls.append(at["prob"]) # model prob/logprob depending on your pipeline

            corr, _ = spearmanr(fre_ls, prob_ls)
            ele["corr"] = float(corr) if corr == corr else 0.0  # handle NaN

            token_alternative_fre_ls[i] = ele

        token_alternative_fre_ls = sorted(token_alternative_fre_ls, key=lambda x: x["corr"], reverse=True)
        return token_alternative_fre_ls
