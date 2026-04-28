
import os

from dataclasses import dataclass
from typing import Optional, Tuple

COMPONENT_MODES = ('just_1', 'all')
K_POLICIES = ('none', 'fixed', 'acc90', 'acc80')


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
    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching", "llamaguard2", "llm_refusal_judge")
    refusal_eval_methodologies: Tuple[str] = ("substring_matching", "llm_refusal_judge")
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048

    # Component selection: 'just_1' hooks only the best (pos, layer); 'all' hooks
    # every layer of the model using the single best direction/P (andyrdt-style).
    component_mode: str = 'just_1'

    # INLP k-regime:
    #   'none'  — full P found by INLP
    #   'fixed' — top ``inlp_k_restrict`` directions via SVD of (I - P)
    #   'acc90' — per-(pos, layer) k = # classifiers with dev acc >= 0.90
    #   'acc80' — per-(pos, layer) k = # classifiers with dev acc >= 0.80
    inlp_k_policy: str = 'none'
    inlp_k_restrict: Optional[int] = None

    # Reflection alphas; alpha=1 recovers plain nullspace projection.
    reflection_alphas: Tuple[float, ...] = (1.0, 2.0)

    # Benchmark evaluation sizes (-1 = all available)
    benchmark_n_mmlu: int = 500
    benchmark_n_arc: int = -1
    benchmark_n_truthfulqa: int = -1

    force_overwrite: bool = False

    def extraction_path(self) -> str:
        """Shared extraction artifacts (dataset splits, mean-diff, INLP results).
        Independent of component_mode / k-regime."""
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias
        )

    def artifact_path(self) -> str:
        """Per-cell artifacts for the mode-specific interventions
        (baseline + ablation + reflection α=1 + α=2)."""
        subdir = f"{self.component_mode}__{self.k_label()}"
        return os.path.join(self.extraction_path(), subdir)

    def actadd_path(self) -> str:
        """Shared-across-modes artifacts for actadd interventions
        (baseline + actadd ×{0.5,1,2} + inlp_actadd ×{0.5,1,2})."""
        return os.path.join(self.extraction_path(), f"actadd__{self.k_label()}")

    def k_label(self) -> str:
        policy = (self.inlp_k_policy or 'none').lower()
        if policy == 'none':
            return 'none'
        if policy == 'fixed':
            k = self.inlp_k_restrict
            return f"k{k}" if k is not None else "kna"
        if policy in ('acc90', 'acc80'):
            return policy
        raise ValueError(f"Unknown inlp_k_policy: {self.inlp_k_policy!r}")
