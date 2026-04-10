# Removing Refusal in LLMs: Diff-in-Means vs. Iterative Nullspace Projection

**Content warning**: This repository contains text that is offensive, harmful, or otherwise inappropriate in nature.

This project compares two approaches for extracting and manipulating the refusal direction in Large Language Models: the diff-in-means method from [Arditi et al. (2024)](https://arxiv.org/abs/2406.11717) and Iterative Nullspace Projection (INLP) from [Ravfogel et al. (2020)](https://aclanthology.org/2020.acl-main.647/). The codebase builds on the [original refusal_direction repository](https://github.com/andyrdt/refusal_direction) and extends it with INLP-based direction extraction, additional intervention types, and an expanded evaluation suite.

- [Write-up](https://docs.google.com/document/d/1s5TXaa0ddkekicoAngFqszzOaZjxw7QN/edit?rtpof=true)

## Overview

Diff-in-means extracts a single refusal direction by computing the difference between mean activations on harmful and harmless prompts. INLP takes a different approach: it iteratively trains linear classifiers to distinguish harmful from harmless representations, projecting out each identified direction until no linear classifier can separate the two. This removes a multi-dimensional subspace rather than a single vector, making no assumption about refusal being captured by one direction.

The pipeline implements both methods side by side and compares them across several families of interventions, applied at varying scopes (single component, top 50%, or all components):

- **Directional ablation** (diff-in-means): removes the single refusal direction from activations.
- **Nullspace projection** (INLP): projects activations into the orthogonal complement of all identified refusal directions.
- **Activation addition (ActAdd)**: steers the model by adding or subtracting a direction with configurable multipliers. Supported for both the diff-in-means direction and the first INLP classifier direction, scaled to the same norm for fair comparison.
- **Counterfactual reflection** (INLP): reflects representations across the nullspace with configurable strength to produce counterfactual behavior.

## Evaluation

Each intervention is evaluated on both safety and performance metrics:

- **Refusal score**: fraction of harmful prompts where the model does not refuse (substring matching).
- **Safety score**: fraction of harmful-prompt responses judged unsafe by LlamaGuard 2.
- **Refusal on harmless prompts**: fraction of harmless prompts the model incorrectly refuses.
- **Perplexity**: cross-entropy loss on Pile (general text) and Alpaca (instruction-following).
- **MMLU**: 5-shot accuracy on Massive Multitask Language Understanding.
- **ARC**: 5-shot accuracy on ARC-Challenge.

## Setup

```bash
git clone https://github.com/eliroc98/refusal_direction.git
cd refusal_direction
source setup.sh
```

The setup script will prompt you for a HuggingFace token (required to access gated models) and a Together AI token (used for evaluating jailbreak safety scores). It will then set up a virtual environment and install the required packages.

## Running the pipeline

```bash
python3 -m pipeline.run_pipeline --model_path {model_path}
```

where `{model_path}` is the path to a HuggingFace model (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`).

The pipeline runs the following stages:

1. **Extract** candidate refusal directions via diff-in-means and INLP.
2. **Select** the best direction for each method by scoring candidates on refusal suppression, steering effectiveness, and KL divergence, applied locally at the source component.
3. **Infer** completions over harmful and harmless prompts under each intervention.
4. **Evaluate** CE loss, benchmark accuracy, and jailbreak metrics.

Artifacts are saved under `pipeline/runs/{model_alias}/`.

### Stage-level execution

The pipeline can be split into independent stages, which is useful when GPU memory is limited or when iterating on a specific phase:

- `--extract_only`: run only direction extraction (stage 1).
- `--select_only`: run only component selection from previously extracted artifacts (stage 2).
- `--infer_only`: run only inference and evaluation from previously selected artifacts (stages 3–4).
- `--use_existing`: skip extraction and selection, reuse pre-computed directions.
- `--resume_from_eval`: skip inference, resume from jailbreak evaluation.
- `--skip_eval`: run inference only, skip jailbreak evaluation.

### Other options

- `--top_percentage`: fraction of filtered ranked components to keep for ablation and nullspace projection (default: 1.0).
- `--compare_rankings`: run the expensive all-layer (global) scoring alongside local scoring and compute the Spearman rank correlation between the two.
- `--device`: device for model loading (`auto`, `cuda:0`, `cpu`).
- `--vllm_gpu_memory_utilization`: fraction of GPU memory available to vLLM classifiers (default: 0.9).
