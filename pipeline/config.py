
import os

from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class Config:
    model_alias: str
    model_path: str
    device: str = 'auto'
    vllm_gpu_memory_utilization: float = 0.9
    n_train: int = 128
    n_test: int = 100
    n_val: int = 32
    filter_train: bool = True
    filter_val: bool = True
    evaluation_datasets: Tuple[str] = ("jailbreakbench",)
    max_new_tokens: int = 512
    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching", "llamaguard2")
    refusal_eval_methodologies: Tuple[str] = ("substring_matching",)
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048
    top_percentage: float = 1.0
    just_one: bool = False
    compare_rankings: bool = False
    # Single optimal INLP projection
    inlp_single_optimal: bool = True
    inlp_k_restrict: Optional[int] = None
    # k-regime policy for the INLP nullspace:
    #   'none'  — use the full P found by INLP (current behaviour).
    #   'fixed' — restrict to the top ``inlp_k_restrict`` directions from P.
    #   'acc99' — per-(pos, layer) k = # classifiers with dev acc >= 0.99.
    #   'acc95' — per-(pos, layer) k = # classifiers with dev acc >= 0.95.
    inlp_k_policy: str = 'none'
    # Intervention mode
    intervention_mode: str = 'both'       # 'actadd', 'reflection', or 'both'
    reflection_alphas: Tuple[float, ...] = (1.0, 2.0)
    # Benchmark evaluation sizes (-1 = all available)
    benchmark_n_mmlu: int = 500        # sample for speed (~57 subjects)
    benchmark_n_arc: int = -1          # full ARC-Challenge test (~1172)
    benchmark_n_truthfulqa: int = -1   # full TruthfulQA validation (~817)
    force_overwrite: bool = False        # force regeneration of existing completion files

    def extraction_path(self) -> str:
        """Path for shared extraction artifacts (dataset splits, mean-diff directions, INLP activations).
        These do not depend on top_percentage and are reused across runs."""
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias)

    def artifact_path(self) -> str:
        """Path for per-run artifacts (selected components, completions, loss evals).
        Scoped by top_percentage (or just_one) and INLP k-regime so different runs
        are kept distinct."""
        if self.just_one:
            subdir = "top_just1"
        else:
            subdir = f"top{self.top_percentage:g}"
        subdir += self._k_suffix()
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias, subdir)

    def _k_suffix(self) -> str:
        policy = (self.inlp_k_policy or 'none').lower()
        if policy == 'none':
            return ''
        if policy == 'fixed':
            k = self.inlp_k_restrict if self.inlp_k_restrict is not None else 'na'
            return f"_k{k}"
        if policy in ('acc99', 'acc95'):
            return f"_{policy}"
        raise ValueError(f"Unknown inlp_k_policy: {self.inlp_k_policy!r}")