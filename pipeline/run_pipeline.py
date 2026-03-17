import gc
import torch
import random
import json
import os
import argparse

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_all_direction_ablation_hooks,
    get_all_nullspace_projection_hooks,
    get_all_direction_ablation_hooks_per_layer,
)

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.generate_directions_inlp import (
    generate_directions_inlp,
    compute_inlp_nullspace_projection,
)
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for model loading. Use "auto" to spread across all available GPUs, '
                             'or specify a single device such as "cuda:0" or "cpu". (default: auto)')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.9,
                        help='Fraction of GPU memory vLLM classifiers (LlamaGuard2, HarmBench) may use. '
                             'Lower this if the main model and classifiers compete for memory. (default: 0.9)')
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

    return harmful_train, harmless_train, harmful_val, harmless_val

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save mean-difference candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs

def generate_and_save_inlp_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save INLP-based candidate directions (first classifier direction per layer).

    Activations are cached to disk to avoid re-running the model when computing
    the full nullspace projection in a subsequent step.
    """
    artifact_dir = os.path.join(cfg.artifact_path(), 'generate_directions_inlp')
    os.makedirs(artifact_dir, exist_ok=True)

    inlp_directions = generate_directions_inlp(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=artifact_dir,
        n_classifiers=1,    # one iteration → most discriminative direction per layer
        min_accuracy=0.55,
    )

    torch.save(inlp_directions, os.path.join(artifact_dir, 'inlp_first_directions.pt'))
    return inlp_directions

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, label=''):
    """Select and save the best direction using the standard selection criteria.

    Parameters
    ----------
    label : str
        Optional suffix for artifact filenames (e.g. 'inlp') to keep results
        from different direction-extraction methods separate.
    """
    suffix = f'_{label}' if label else ''
    artifact_dir = os.path.join(cfg.artifact_path(), f'select_direction{suffix}')
    os.makedirs(artifact_dir, exist_ok=True)

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=artifact_dir,
    )

    with open(f'{cfg.artifact_path()}/direction_metadata{suffix}.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction{suffix}.pt')

    return pos, layer, direction

def compute_and_save_nullspace_projection(cfg, model_base, harmful_train, harmless_train, pos, layer, n_classifiers=20):
    """Compute the INLP nullspace projection matrix for the selected (pos, layer).

    Loads cached activations from the INLP direction generation step.
    """
    artifact_dir = os.path.join(cfg.artifact_path(), 'generate_directions_inlp')
    P = compute_inlp_nullspace_projection(
        artifact_dir=artifact_dir,
        model_base=model_base,
        pos=pos,
        layer=layer,
        n_classifiers=n_classifiers,
        min_accuracy=0.55,
    )
    import numpy as np
    np.save(os.path.join(cfg.artifact_path(), 'nullspace_projection.npy'), P)
    return P

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
    )

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

def run_pipeline(model_path, device='auto', vllm_gpu_memory_utilization=0.9):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, device=device,
                 vllm_gpu_memory_utilization=vllm_gpu_memory_utilization)

    model_base = construct_model_base(cfg.model_path, device=cfg.device)

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)

    # For steps 1 and 2, direction selection uses the nullspace projection's first
    # classifier direction (not the full P), so both mean-diff and INLP share the
    # same select_direction() logic.

    # 1. Generate candidate refusal directions
    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)

    # 1b. Generate candidate directions via INLP (discriminatively trained)
    inlp_directions = generate_and_save_inlp_directions(cfg, model_base, harmful_train, harmless_train)

    # 2. Select the most effective mean-difference direction
    pos, layer, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions)

    # 2b. Select the most effective INLP direction (same selection framework)
    inlp_pos, inlp_layer, inlp_direction = select_and_save_direction(
        cfg, model_base, harmful_val, harmless_val, inlp_directions, label='inlp'
    )

    # 2c. Compute full INLP nullspace projection for the selected INLP (pos, layer)
    #     This removes ALL iteratively found refusal directions, not just the first.
    P = compute_and_save_nullspace_projection(
        cfg, model_base, harmful_train, harmless_train, inlp_pos, inlp_layer, n_classifiers=20
    )

    # ── Build intervention hooks ───────────────────────────────────────────────

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []

    # Mean-difference direction: ablate across all layers / actadd at selected layer
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

    # INLP nullspace projection: removes the entire refusal subspace (all layers)
    nullspace_fwd_pre_hooks, nullspace_fwd_hooks = get_all_nullspace_projection_hooks(model_base, P)

    # INLP first direction: actadd at the selected INLP layer (coeff=-1 to jailbreak)
    inlp_actadd_fwd_pre_hooks, inlp_actadd_fwd_hooks = [
        (model_base.model_block_modules[inlp_layer],
         get_activation_addition_input_pre_hook(vector=inlp_direction, coeff=-1.0))
    ], []

    # Component-specific (per-layer) interventions: each layer uses its own direction
    ablation_per_layer_fwd_pre_hooks, ablation_per_layer_fwd_hooks = \
        get_all_direction_ablation_hooks_per_layer(model_base, candidate_directions, pos)
    inlp_ablation_per_layer_fwd_pre_hooks, inlp_ablation_per_layer_fwd_hooks = \
        get_all_direction_ablation_hooks_per_layer(model_base, inlp_directions, inlp_pos)

    # 3a. Generate and save completions on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', dataset_name)

        # Nullspace projection intervention (removes full INLP refusal subspace)
        generate_and_save_completions_for_dataset(cfg, model_base, nullspace_fwd_pre_hooks, nullspace_fwd_hooks, 'nullspace', dataset_name)

        # INLP first-direction actadd (discriminatively trained direction)
        generate_and_save_completions_for_dataset(cfg, model_base, inlp_actadd_fwd_pre_hooks, inlp_actadd_fwd_hooks, 'inlp_actadd', dataset_name)

        # Component-specific (per-layer) ablation with mean-diff and INLP directions
        generate_and_save_completions_for_dataset(cfg, model_base, ablation_per_layer_fwd_pre_hooks, ablation_per_layer_fwd_hooks, 'ablation_per_layer', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, inlp_ablation_per_layer_fwd_pre_hooks, inlp_ablation_per_layer_fwd_hooks, 'inlp_ablation_per_layer', dataset_name)

    # 4a. Generate and save completions on harmless evaluation dataset
    #     (We test whether each intervention INDUCES refusal on harmless prompts,
    #      which would indicate the intervention is not perfectly specific to harmful
    #      instructions.  Ablation-style hooks are omitted here because removing the
    #      refusal direction on already-harmless prompts is not a refusal-induction test.)
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)

    # Mean-diff actadd with coeff=+1.0: add refusal direction to check induction
    actadd_refusal_pre_hooks, actadd_refusal_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))], []
    generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, 'actadd', 'harmless', dataset=harmless_test)

    # INLP actadd with coeff=+1.0: add INLP direction to check refusal induction
    inlp_actadd_refusal_pre_hooks, inlp_actadd_refusal_hooks = [
        (model_base.model_block_modules[inlp_layer],
         get_activation_addition_input_pre_hook(vector=inlp_direction, coeff=+1.0))
    ], []
    generate_and_save_completions_for_dataset(cfg, model_base, inlp_actadd_refusal_pre_hooks, inlp_actadd_refusal_hooks, 'inlp_actadd', 'harmless', dataset=harmless_test)

    # 5. Evaluate loss on harmless datasets for all interventions
    #    (checks whether interventions degrade performance on benign prompts)
    evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
    evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')
    evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd')

    # Nullspace projection loss (should have low impact on harmless perplexity)
    evaluate_loss_for_datasets(cfg, model_base, nullspace_fwd_pre_hooks, nullspace_fwd_hooks, 'nullspace')

    # INLP actadd loss
    evaluate_loss_for_datasets(cfg, model_base, inlp_actadd_fwd_pre_hooks, inlp_actadd_fwd_hooks, 'inlp_actadd')

    # Component-specific ablation losses
    evaluate_loss_for_datasets(cfg, model_base, ablation_per_layer_fwd_pre_hooks, ablation_per_layer_fwd_hooks, 'ablation_per_layer')
    evaluate_loss_for_datasets(cfg, model_base, inlp_ablation_per_layer_fwd_pre_hooks, inlp_ablation_per_layer_fwd_hooks, 'inlp_ablation_per_layer')

    # Free model_base from GPU before loading LlamaGuard2 for evaluation
    model_base.del_model()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3b. Evaluate completions and save results on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

        # Nullspace projection evaluation
        evaluate_completions_and_save_results_for_dataset(cfg, 'nullspace', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

        # INLP actadd evaluation
        evaluate_completions_and_save_results_for_dataset(cfg, 'inlp_actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

        # Component-specific evaluations
        evaluate_completions_and_save_results_for_dataset(cfg, 'ablation_per_layer', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'inlp_ablation_per_layer', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

    # 4b. Evaluate completions and save results on harmless evaluation dataset
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)

    # INLP actadd harmless evaluation
    evaluate_completions_and_save_results_for_dataset(cfg, 'inlp_actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, device=args.device,
                 vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization)
