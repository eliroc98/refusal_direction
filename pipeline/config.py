
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
    compare_rankings: bool = False
    # Single optimal INLP projection
    inlp_single_optimal: bool = True
    inlp_k_restrict: Optional[int] = None
    # Intervention mode
    intervention_mode: str = 'both'       # 'actadd', 'reflection', or 'both'
    reflection_alphas: Tuple[float, ...] = (1.0, 2.0)
    # Benchmark evaluation sizes (-1 = all available)
    benchmark_n_mmlu: int = 500        # sample for speed (~57 subjects)
    benchmark_n_arc: int = -1          # full ARC-Challenge test (~1172)
    benchmark_n_truthfulqa: int = -1   # full TruthfulQA validation (~817)

    def extraction_path(self) -> str:
        """Path for shared extraction artifacts (dataset splits, mean-diff directions, INLP activations).
        These do not depend on top_percentage and are reused across runs."""
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias)

    def artifact_path(self) -> str:
        """Path for per-run artifacts (selected components, completions, loss evals).
        Scoped by top_percentage so different runs are kept distinct."""
        top_pct_str = f"top{self.top_percentage:g}"
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias, top_pct_str)