import gc
import torch
import random
import json
import os
import argparse
import math

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
    get_nullspace_projection_input_pre_hook,
    get_nullspace_projection_output_hook,
    get_direction_ablation_hooks,
    get_nullspace_projection_hooks,
)

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.generate_directions_inlp import (
    generate_directions_inlp,
    select_direction_inlp_ranked,
)
from pipeline.submodules.select_direction import select_direction_ranked, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

ACTADD_TARGET_MULTIPLIERS = [0.5, 1.0, 2.0]

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
    parser.add_argument('--resume_from_eval', action='store_true',
                        help='Skip model inference (steps 1-5) and resume from LlamaGuard evaluation. '
                             'Assumes completions have already been generated in a previous run.')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Run only the inference steps (1-5) and skip the LlamaGuard evaluation. '
                             'Use --resume_from_eval in a separate process to run evaluation afterwards, '
                             'freeing GPU memory between the two phases.')
    parser.add_argument('--use_existing', action='store_true',
                        help='Skip direction extraction (steps 1-2) and load pre-computed directions '
                             'from a previous run (direction.pt, direction_inlp.pt, etc). '
                             'Re-runs the full intervention sweep (steps 3-5) with normalized '
                             'directions and coefficient sweep, then evaluation.')
    parser.add_argument('--top_percentage', type=float, default=1.0,
                        help='Top percentage of filtered ranked components to keep for both mean-diff '
                            'ablation and INLP nullspace projection. Final selected count is shared '
                            'across methods as min(target_ablation, target_inlp). (default: 1.0)')
    parser.add_argument('--extract_only', action='store_true',
                        help='Run only extraction artifacts (sampling/filtering + mean-diff + INLP activations).')
    parser.add_argument('--select_only', action='store_true',
                        help='Run only component selection from previously extracted artifacts.')
    parser.add_argument('--infer_only', action='store_true',
                        help='Run only inference/loss from previously selected artifacts.')
    parser.add_argument('--compare_rankings', action='store_true',
                        help='Run expensive all-layer (global) scoring and Spearman rank '
                             'comparison against local (per-component) ranking.')
    return parser.parse_args()

def save_run_params(cfg, extra_flags=None):
    """Save all run parameters to run_params.json in the artifact directory."""
    import dataclasses
    os.makedirs(cfg.artifact_path(), exist_ok=True)
    params = dataclasses.asdict(cfg)
    if extra_flags:
        params.update(extra_flags)
    with open(os.path.join(cfg.artifact_path(), 'run_params.json'), 'w') as f:
        json.dump(params, f, indent=2)


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


def save_dataset_artifacts(cfg, harmful_train, harmless_train, harmful_val, harmless_val):
    """Persist sampled/filtered datasets so selection can run independently later."""
    payload = {
        "seed": 42,
        "harmful_train": harmful_train,
        "harmless_train": harmless_train,
        "harmful_val": harmful_val,
        "harmless_val": harmless_val,
        "counts": {
            "harmful_train": len(harmful_train),
            "harmless_train": len(harmless_train),
            "harmful_val": len(harmful_val),
            "harmless_val": len(harmless_val),
        },
    }
    with open(os.path.join(cfg.extraction_path(), 'dataset_artifacts.json'), 'w') as f:
        json.dump(payload, f, indent=2)


def load_dataset_artifacts(cfg):
    path = os.path.join(cfg.extraction_path(), 'dataset_artifacts.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing dataset artifacts at {path}. Run extraction first (or a full pipeline run)."
        )
    with open(path, 'r') as f:
        payload = json.load(f)

    return (
        payload["harmful_train"],
        payload["harmless_train"],
        payload["harmful_val"],
        payload["harmless_val"],
    )

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
    if not os.path.exists(os.path.join(cfg.extraction_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.extraction_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,)

    torch.save(mean_diffs, os.path.join(cfg.extraction_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs

def generate_and_save_inlp_directions(cfg, model_base, harmful_train, harmless_train):
    """Compute and save INLP parameters (P, first_dir, accuracies) for all (pos, layer) pairs."""
    artifact_dir = os.path.join(cfg.extraction_path(), 'generate_directions_inlp')
    generate_directions_inlp(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=artifact_dir,
    )

def _resolve_target_count(pool_size, top_percentage):
    if pool_size <= 0:
        return 0
    pct = max(0.0, min(100.0, float(top_percentage)))
    return max(1, int(math.ceil(pool_size * (pct / 100.0))))


def _make_ablation_component(row, candidate_directions):
    """Convert a scored row into a component dict with its direction tensor."""
    pos = int(row['position'])
    layer = int(row['layer'])
    comp = {
        'position': pos,
        'layer': layer,
        'refusal_score_local': float(row['refusal_score_local']),
        'steering_median_score_local': float(row['steering_median_score_local']),
        'kl_div_score_local': float(row['kl_div_score_local']),
        'sorting_score_local': float(row['sorting_score_local']),
        'direction': candidate_directions[pos, layer].detach().cpu().float(),
    }
    # Global scores may be NaN when compare_rankings is False
    for key in ('refusal_score', 'steering_median_score', 'kl_div_score', 'sorting_score'):
        comp[key] = float(row.get(key, float('nan')))
    return comp


def select_ranked_direction_components(cfg, model_base, harmful_val, harmless_val, candidate_directions, actadd_multipliers):
    artifact_dir = os.path.join(cfg.extraction_path(), 'select_direction')
    os.makedirs(artifact_dir, exist_ok=True)

    all_ranked, filtered_ranked, top_direction_norm = select_direction_ranked(
        model_base=model_base,
        harmful_instructions=harmful_val,
        harmless_instructions=harmless_val,
        candidate_directions=candidate_directions,
        artifact_dir=artifact_dir,
        actadd_multipliers=actadd_multipliers,
        compare_rankings=cfg.compare_rankings,
    )

    all_components = [_make_ablation_component(row, candidate_directions) for row in all_ranked]
    filtered_components = [_make_ablation_component(row, candidate_directions) for row in filtered_ranked]

    return all_components, filtered_components, top_direction_norm


def _make_inlp_component(row):
    """Convert an INLP scored row into a component dict."""
    return {
        'position': int(row['position']),
        'layer': int(row['layer']),
        'refusal_score': float(row['refusal_score']),
        'steering_score': float(row['steering_score']),
        'kl_div_score': float(row['kl_div_score']),
        'sorting_score': float(row['sorting_score']),
        'direction': torch.from_numpy(row['first_dir']).float(),
        'P': row['P'],
    }


def select_ranked_inlp_components(cfg, model_base, harmful_val, harmless_val, actadd_multipliers, direction_norm):
    artifact_dir = os.path.join(cfg.extraction_path(), 'generate_directions_inlp')

    all_ranked, filtered_ranked = select_direction_inlp_ranked(
        artifact_dir=artifact_dir,
        model_base=model_base,
        harmful_instructions=harmful_val,
        harmless_instructions=harmless_val,
        actadd_multipliers=actadd_multipliers,
        direction_norm=direction_norm,
    )

    all_components = [_make_inlp_component(row) for row in all_ranked]
    filtered_components = [_make_inlp_component(row) for row in filtered_ranked]

    return all_components, filtered_components


def _safe_float(val, default=None):
    """Return val as float, replacing NaN with default for JSON safety."""
    v = float(val)
    return default if math.isnan(v) else v


def save_selected_component_artifacts(cfg, selected_ablation, selected_inlp,
                                      ranked_ablation, ranked_inlp,
                                      filtered_ablation, filtered_inlp,
                                      ablation_pool_size, inlp_pool_size,
                                      shared_count, target_ablation, target_inlp):
    import numpy as np

    extraction_path = cfg.extraction_path()
    artifact_path = cfg.artifact_path()
    os.makedirs(artifact_path, exist_ok=True)

    # Persist complete ranked pools (unfiltered) for later reselection — shared across runs.
    torch.save(ranked_ablation, os.path.join(extraction_path, 'ablation_components_ranked.pt'))
    torch.save(ranked_inlp, os.path.join(extraction_path, 'inlp_components_ranked.pt'))

    # Persist filtered pools for best-direction recovery — shared across runs.
    torch.save(filtered_ablation, os.path.join(extraction_path, 'ablation_components_filtered.pt'))
    torch.save(filtered_inlp, os.path.join(extraction_path, 'inlp_components_filtered.pt'))

    with open(os.path.join(artifact_path, 'selected_components_metadata.json'), 'w') as f:
        json.dump({
            'top_percentage': cfg.top_percentage,
            'ablation_pool_size': ablation_pool_size,
            'inlp_pool_size': inlp_pool_size,
            'target_ablation': target_ablation,
            'target_inlp': target_inlp,
            'shared_count': shared_count,
            'best_ablation': {
                'position': filtered_ablation[0]['position'],
                'layer': filtered_ablation[0]['layer'],
            },
            'best_inlp': {
                'position': filtered_inlp[0]['position'],
                'layer': filtered_inlp[0]['layer'],
            } if filtered_inlp else None,
            'selected_ablation': [
                {
                    'position': c['position'],
                    'layer': c['layer'],
                    'refusal_score_local': _safe_float(c.get('refusal_score_local', float('nan'))),
                    'steering_median_score_local': _safe_float(c.get('steering_median_score_local', float('nan'))),
                    'kl_div_score_local': _safe_float(c.get('kl_div_score_local', float('nan'))),
                }
                for c in selected_ablation
            ],
            'selected_inlp': [
                {
                    'position': c['position'],
                    'layer': c['layer'],
                    'refusal_score': _safe_float(c.get('refusal_score', float('nan'))),
                    'steering_score': _safe_float(c.get('steering_score', float('nan'))),
                    'kl_div_score': _safe_float(c.get('kl_div_score', float('nan'))),
                }
                for c in selected_inlp
            ],
        }, f, indent=4)

    # Backward-compatible single-component files use the best from filtered pool — shared across runs.
    best_ablation = filtered_ablation[0]

    with open(os.path.join(extraction_path, 'direction_metadata.json'), 'w') as f:
        json.dump({'pos': best_ablation['position'], 'layer': best_ablation['layer']}, f, indent=4)
    torch.save(best_ablation['direction'], os.path.join(extraction_path, 'direction.pt'))

    if filtered_inlp:
        best_inlp = filtered_inlp[0]
        with open(os.path.join(extraction_path, 'direction_metadata_inlp.json'), 'w') as f:
            json.dump({'pos': best_inlp['position'], 'layer': best_inlp['layer']}, f, indent=4)
        torch.save(best_inlp['direction'], os.path.join(extraction_path, 'direction_inlp.pt'))
        np.save(os.path.join(extraction_path, 'nullspace_projection.npy'), best_inlp['P'])


def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies, llamaguard2_classifier=None):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        llamaguard2_classifier=llamaguard2_classifier,
    )

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path, intervention_label=intervention_label)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

def _run_inference(cfg, model_path):
    """Steps 1-5: model loading, direction extraction, completions, and loss evaluation."""
    model_base = construct_model_base(cfg.model_path, device=cfg.device)

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)

    # Persist filtered sampled datasets so selection-only can reuse exactly this split.
    save_dataset_artifacts(cfg, harmful_train, harmless_train, harmful_val, harmless_val)

    # 1. Generate candidate refusal directions
    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)

    # 1b. Generate candidate directions via INLP (discriminatively trained)
    generate_and_save_inlp_directions(cfg, model_base, harmful_train, harmless_train)

    # 2. Rank candidate components for mean-diff and INLP with shared criteria.
    actadd_multipliers = ACTADD_TARGET_MULTIPLIERS

    all_ablation, filtered_ablation, top_direction_norm = select_ranked_direction_components(
        cfg, model_base, harmful_val, harmless_val, candidate_directions, actadd_multipliers
    )
    all_inlp, filtered_inlp = select_ranked_inlp_components(
        cfg, model_base, harmful_val, harmless_val, actadd_multipliers, top_direction_norm
    )

    # Top-k layers from unfiltered pool
    n_components = _resolve_target_count(len(all_ablation), cfg.top_percentage)

    selected_ablation_layers = all_ablation[:n_components]
    selected_inlp_layers = all_inlp[:n_components]

    save_selected_component_artifacts(
        cfg,
        selected_ablation=selected_ablation_layers,
        selected_inlp=selected_inlp_layers,
        ranked_ablation=all_ablation,
        ranked_inlp=all_inlp,
        filtered_ablation=filtered_ablation,
        filtered_inlp=filtered_inlp,
        ablation_pool_size=len(all_ablation),
        inlp_pool_size=len(all_inlp),
        shared_count=n_components,
        target_ablation=target_ablation,
        target_inlp=target_inlp,
    )

    print(
        f"Selected shared component count={n_components} "
        f"(ablation target={target_ablation}/{len(all_ablation)}, "
        f"inlp target={target_inlp}/{len(all_inlp)}, "
        f"top_percentage={cfg.top_percentage:.3f}%)"
    )

    assert n_components > 0, "No components selected. Adjust top_percentage or check filtering criteria."

    # Best direction/layer from filtered pool (for ablation direction + actadd)
    best_direction = filtered_ablation[0]['direction']
    best_layer = filtered_ablation[0]['layer']

    assert len(filtered_inlp) > 0, "No INLP components selected. Adjust top_percentage or check filtering criteria."

    best_inlp_direction = filtered_inlp[0]['direction']
    best_inlp_layer = filtered_inlp[0]['layer']

    # -- Normalize directions for fair comparison ---------------------------------
    direction_norm = torch.norm(best_direction).item()
    direction_unit = best_direction / (torch.norm(best_direction) + 1e-8)
    inlp_direction_unit = best_inlp_direction / (torch.norm(best_inlp_direction) + 1e-8)


    actadd_multipliers = ACTADD_TARGET_MULTIPLIERS
    actadd_coeffs = [m * direction_norm for m in actadd_multipliers]
    print(f"Direction norm (diff-in-means): {direction_norm:.4f}")
    print(f"ActAdd coefficient sweep: {[f'{c:.2f}' for c in actadd_coeffs]}")

    # Persist coefficients so _run_evaluation can discover them
    with open(os.path.join(cfg.extraction_path(), 'actadd_coeffs.json'), 'w') as f:
        json.dump({"direction_norm": direction_norm, "multipliers": actadd_multipliers, "coeffs": actadd_coeffs}, f, indent=2)

    # -- Build intervention hooks --------------------------------------------------

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []

    # Ablation: best direction (filtered rank-1) at top-k layers (unfiltered)
    # Nullspace: per-component P at top-k layers (unfiltered)
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_direction_ablation_hooks(
        model_base, selected_ablation_layers, best_direction)
    nullspace_fwd_pre_hooks, nullspace_fwd_hooks = get_nullspace_projection_hooks(
        model_base, selected_inlp_layers)

    # 3a. Generate and save completions on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)

        generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, nullspace_fwd_pre_hooks, nullspace_fwd_hooks, 'nullspace', dataset_name)

        # Sweep coefficients for mean-diff actadd (and INLP actadd when available)
        for coeff in actadd_coeffs:
            label = f'actadd_c{coeff:.2f}'
            hooks_pre = [(model_base.model_block_modules[best_layer],
                          get_activation_addition_input_pre_hook(vector=direction_unit, coeff=-coeff))]
            generate_and_save_completions_for_dataset(cfg, model_base, hooks_pre, [], label, dataset_name)

            inlp_label = f'inlp_actadd_c{coeff:.2f}'
            inlp_hooks_pre = [(model_base.model_block_modules[best_inlp_layer],
                                get_activation_addition_input_pre_hook(vector=inlp_direction_unit, coeff=-coeff))]
            generate_and_save_completions_for_dataset(cfg, model_base, inlp_hooks_pre, [], inlp_label, dataset_name)

    # 4a. Generate and save completions on harmless evaluation dataset
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)

    # Sweep coefficients: add refusal direction (+coeff) to harmless prompts
    for coeff in actadd_coeffs:
        label = f'actadd_c{coeff:.2f}'
        hooks_pre = [(model_base.model_block_modules[best_layer],
                      get_activation_addition_input_pre_hook(vector=direction_unit, coeff=+coeff))]
        generate_and_save_completions_for_dataset(cfg, model_base, hooks_pre, [], label, 'harmless', dataset=harmless_test)

        inlp_label = f'inlp_actadd_c{coeff:.2f}'
        inlp_hooks_pre = [(model_base.model_block_modules[best_inlp_layer],
                            get_activation_addition_input_pre_hook(vector=inlp_direction_unit, coeff=+coeff))]
        generate_and_save_completions_for_dataset(cfg, model_base, inlp_hooks_pre, [], inlp_label, 'harmless', dataset=harmless_test)

    # 5. Evaluate loss on harmless datasets for all interventions
    evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
    evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')
    evaluate_loss_for_datasets(cfg, model_base, nullspace_fwd_pre_hooks, nullspace_fwd_hooks, 'nullspace')

    for coeff in actadd_coeffs:
        label = f'actadd_c{coeff:.2f}'
        hooks_pre = [(model_base.model_block_modules[best_layer],
                      get_activation_addition_input_pre_hook(vector=direction_unit, coeff=-coeff))]
        evaluate_loss_for_datasets(cfg, model_base, hooks_pre, [], label)

        inlp_label = f'inlp_actadd_c{coeff:.2f}'
        inlp_hooks_pre = [(model_base.model_block_modules[best_inlp_layer],
                            get_activation_addition_input_pre_hook(vector=inlp_direction_unit, coeff=-coeff))]
        evaluate_loss_for_datasets(cfg, model_base, inlp_hooks_pre, [], inlp_label)

    # Free ALL GPU resources before loading LlamaGuard2 for evaluation.
    model_base.del_model()
    del model_base
    del candidate_directions, best_direction, direction_unit
    if has_inlp:
        del best_inlp_direction, inlp_direction_unit
    del all_ablation, all_inlp, filtered_ablation, filtered_inlp
    del selected_ablation_layers, selected_inlp_layers
    del baseline_fwd_pre_hooks, baseline_fwd_hooks
    del ablation_fwd_pre_hooks, ablation_fwd_hooks
    del nullspace_fwd_pre_hooks, nullspace_fwd_hooks

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_extraction(cfg, model_path):
    """Run only extraction artifacts: datasets + candidate mean-diff + INLP activations."""
    model_base = construct_model_base(cfg.model_path, device=cfg.device)

    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val
    )

    save_dataset_artifacts(cfg, harmful_train, harmless_train, harmful_val, harmless_val)
    generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)
    generate_and_save_inlp_directions(cfg, model_base, harmful_train, harmless_train)

    model_base.del_model()
    del model_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_selection(cfg, model_path):
    """Run only component selection using pre-extracted artifacts."""
    model_base = construct_model_base(cfg.model_path, device=cfg.device)
    _, _, harmful_val, harmless_val = load_dataset_artifacts(cfg)

    mean_diffs_path = os.path.join(cfg.extraction_path(), 'generate_directions', 'mean_diffs.pt')
    if not os.path.exists(mean_diffs_path):
        raise FileNotFoundError(
            f"Missing extracted mean-diff artifacts at {mean_diffs_path}. Run --extract_only first."
        )
    candidate_directions = torch.load(mean_diffs_path, map_location='cpu', weights_only=True)

    actadd_multipliers = ACTADD_TARGET_MULTIPLIERS

    all_ablation, filtered_ablation, top_direction_norm = select_ranked_direction_components(
        cfg, model_base, harmful_val, harmless_val, candidate_directions, actadd_multipliers
    )
    all_inlp, filtered_inlp = select_ranked_inlp_components(
        cfg, model_base, harmful_val, harmless_val, actadd_multipliers, top_direction_norm
    )

    n_components = _resolve_target_count(len(all_ablation), cfg.top_percentage)

    selected_ablation_layers = all_ablation[:n_components]
    selected_inlp_layers = all_inlp[:n_components]

    save_selected_component_artifacts(
        cfg,
        selected_ablation=selected_ablation_layers,
        selected_inlp=selected_inlp_layers,
        ranked_ablation=all_ablation,
        ranked_inlp=all_inlp,
        filtered_ablation=filtered_ablation,
        filtered_inlp=filtered_inlp,
        ablation_pool_size=len(all_ablation),
        inlp_pool_size=len(all_inlp),
        shared_count=n_components,
        target_ablation=target_ablation,
        target_inlp=target_inlp,
    )

    print(
        f"Selection complete with n_components={n_components} "
        f"(ablation target={target_ablation}/{len(all_ablation)}, "
        f"inlp target={target_inlp}/{len(all_inlp)}, "
        f"top_percentage={cfg.top_percentage:.3f}%)"
    )

    model_base.del_model()
    del model_base
    del candidate_directions, all_ablation, all_inlp, filtered_ablation, filtered_inlp
    del selected_ablation_layers, selected_inlp_layers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_evaluation(cfg):
    """Steps 3b/4b: LlamaGuard evaluation on previously generated completions."""
    # Pin vllm to the same GPU used by the main model so it doesn't default to GPU 0.
    if cfg.device not in ('auto', 'cpu'):
        gpu_index = cfg.device.split(':')[-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index

    # Create the LlamaGuard2 classifier once and reuse it for all evaluations
    # to avoid repeated 15GB model loading/unloading and OOM from memory fragmentation.
    from pipeline.submodules.evaluate_jailbreak import LlamaGuard2Classifier
    lg2_classifier = None
    if "llamaguard2" in cfg.jailbreak_eval_methodologies:
        lg2_classifier = LlamaGuard2Classifier(gpu_memory_utilization=cfg.vllm_gpu_memory_utilization)

    # Load coefficient sweep from inference stage
    coeffs_path = os.path.join(cfg.extraction_path(), 'actadd_coeffs.json')
    with open(coeffs_path, 'r') as f:
        actadd_coeffs = json.load(f)["coeffs"]

    # 3b. Evaluate completions and save results on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, llamaguard2_classifier=lg2_classifier)
        evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, llamaguard2_classifier=lg2_classifier)

        # Nullspace projection evaluation
        evaluate_completions_and_save_results_for_dataset(cfg, 'nullspace', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, llamaguard2_classifier=lg2_classifier)

        # Sweep coefficients for mean-diff and INLP actadd
        for coeff in actadd_coeffs:
            evaluate_completions_and_save_results_for_dataset(cfg, f'actadd_c{coeff:.2f}', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, llamaguard2_classifier=lg2_classifier)
            evaluate_completions_and_save_results_for_dataset(cfg, f'inlp_actadd_c{coeff:.2f}', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, llamaguard2_classifier=lg2_classifier)

    # 4b. Evaluate completions and save results on harmless evaluation dataset
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)

    for coeff in actadd_coeffs:
        evaluate_completions_and_save_results_for_dataset(cfg, f'actadd_c{coeff:.2f}', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, f'inlp_actadd_c{coeff:.2f}', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)

    # Clean up the LlamaGuard2 classifier
    if lg2_classifier is not None:
        lg2_classifier.cleanup()


def _run_inference_from_existing(cfg, model_path):
    """Re-run interventions (steps 3-5) using pre-computed artifacts."""
    import numpy as np

    artifact_path = cfg.artifact_path()
    extraction_path = cfg.extraction_path()

    selected_meta_path = os.path.join(artifact_path, 'selected_components_metadata.json')
    ranked_ablation_path = os.path.join(extraction_path, 'ablation_components_ranked.pt')
    ranked_inlp_path = os.path.join(extraction_path, 'inlp_components_ranked.pt')
    filtered_ablation_path = os.path.join(extraction_path, 'ablation_components_filtered.pt')
    filtered_inlp_path = os.path.join(extraction_path, 'inlp_components_filtered.pt')

    selected_ablation_layers = None
    selected_inlp_layers = None
    best_direction = None
    best_layer = None
    best_inlp_direction = None
    best_inlp_layer = None
    shared_count = 0

    ranked_ablation = torch.load(ranked_ablation_path, map_location='cpu', weights_only=False)
    ranked_inlp = torch.load(ranked_inlp_path, map_location='cpu', weights_only=False)

    shared_count = _resolve_target_count(len(ranked_ablation), cfg.top_percentage)
    assert shared_count > 0, "No components selected. Adjust top_percentage or check filtering criteria."
    if len(ranked_ablation) < shared_count or len(ranked_inlp) < shared_count:
        raise RuntimeError(
            "Existing multi-component artifacts are insufficient for requested top_percentage. "
            "Please rerun extraction/selection without --use_existing."
        )

    selected_ablation_layers = ranked_ablation[:shared_count]
    selected_inlp_layers = ranked_inlp[:shared_count]

    # Best direction from filtered pool
    filtered_ablation = torch.load(filtered_ablation_path, map_location='cpu', weights_only=False)
    filtered_inlp = torch.load(filtered_inlp_path, map_location='cpu', weights_only=False)
    best_direction = filtered_ablation[0]['direction']
    best_layer = filtered_ablation[0]['layer']
    assert len(filtered_inlp)>0, "No INLP components in filtered pool. Cannot recover best INLP direction/layer."

    best_inlp_direction = filtered_inlp[0]['direction']
    best_inlp_layer = filtered_inlp[0]['layer']

    print(
        f"Loaded ranked multi-component artifacts from {extraction_path} with shared_count={shared_count} "
        f"(targets: ablation={target_ablation}, inlp={target_inlp})."
    )
    # --- Load model ---
    model_base = construct_model_base(cfg.model_path, device=cfg.device)

    # -- Normalize directions for fair comparison ---------------------------------
    direction_norm = torch.norm(best_direction).item()
    direction_unit = best_direction / (torch.norm(best_direction) + 1e-8)
    inlp_direction_unit = best_inlp_direction / (torch.norm(best_inlp_direction) + 1e-8)


    actadd_multipliers = ACTADD_TARGET_MULTIPLIERS
    actadd_coeffs = [m * direction_norm for m in actadd_multipliers]
    print(f"Direction norm (diff-in-means): {direction_norm:.4f}")
    print(f"ActAdd coefficient sweep: {[f'{c:.2f}' for c in actadd_coeffs]}")

    # Persist coefficients so _run_evaluation can discover them
    with open(os.path.join(cfg.extraction_path(), 'actadd_coeffs.json'), 'w') as f:
        json.dump({"direction_norm": direction_norm, "multipliers": actadd_multipliers, "coeffs": actadd_coeffs}, f, indent=2)

    # -- Build intervention hooks --------------------------------------------------

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []

    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_direction_ablation_hooks(
        model_base, selected_ablation_layers, best_direction)
    nullspace_fwd_pre_hooks, nullspace_fwd_hooks = get_nullspace_projection_hooks(
        model_base, selected_inlp_layers)

    # 3a. Generate and save completions on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)

        if shared_count > 0:
            generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name)
            generate_and_save_completions_for_dataset(cfg, model_base, nullspace_fwd_pre_hooks, nullspace_fwd_hooks, 'nullspace', dataset_name)

        # Sweep coefficients for mean-diff actadd (and INLP actadd when available)
        for coeff in actadd_coeffs:
            label = f'actadd_c{coeff:.2f}'
            hooks_pre = [(model_base.model_block_modules[best_layer],
                          get_activation_addition_input_pre_hook(vector=direction_unit, coeff=-coeff))]
            generate_and_save_completions_for_dataset(cfg, model_base, hooks_pre, [], label, dataset_name)

            inlp_label = f'inlp_actadd_c{coeff:.2f}'
            inlp_hooks_pre = [(model_base.model_block_modules[best_inlp_layer],
                                get_activation_addition_input_pre_hook(vector=inlp_direction_unit, coeff=-coeff))]
            generate_and_save_completions_for_dataset(cfg, model_base, inlp_hooks_pre, [], inlp_label, dataset_name)

    # 4a. Generate and save completions on harmless evaluation dataset
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)

    # Sweep coefficients: add refusal direction (+coeff) to harmless prompts
    for coeff in actadd_coeffs:
        label = f'actadd_c{coeff:.2f}'
        hooks_pre = [(model_base.model_block_modules[best_layer],
                      get_activation_addition_input_pre_hook(vector=direction_unit, coeff=+coeff))]
        generate_and_save_completions_for_dataset(cfg, model_base, hooks_pre, [], label, 'harmless', dataset=harmless_test)

        inlp_label = f'inlp_actadd_c{coeff:.2f}'
        inlp_hooks_pre = [(model_base.model_block_modules[best_inlp_layer],
                            get_activation_addition_input_pre_hook(vector=inlp_direction_unit, coeff=+coeff))]
        generate_and_save_completions_for_dataset(cfg, model_base, inlp_hooks_pre, [], inlp_label, 'harmless', dataset=harmless_test)

    # 5. Evaluate loss on harmless datasets for all interventions
    evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
    evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')
    evaluate_loss_for_datasets(cfg, model_base, nullspace_fwd_pre_hooks, nullspace_fwd_hooks, 'nullspace')

    for coeff in actadd_coeffs:
        label = f'actadd_c{coeff:.2f}'
        hooks_pre = [(model_base.model_block_modules[best_layer],
                      get_activation_addition_input_pre_hook(vector=direction_unit, coeff=-coeff))]
        evaluate_loss_for_datasets(cfg, model_base, hooks_pre, [], label)

        inlp_label = f'inlp_actadd_c{coeff:.2f}'
        inlp_hooks_pre = [(model_base.model_block_modules[best_inlp_layer],
                            get_activation_addition_input_pre_hook(vector=inlp_direction_unit, coeff=-coeff))]
        evaluate_loss_for_datasets(cfg, model_base, inlp_hooks_pre, [], inlp_label)

    # Free ALL GPU resources before loading LlamaGuard2 for evaluation.
    model_base.del_model()
    del model_base
    del best_direction, direction_unit
    if has_inlp:
        del best_inlp_direction, inlp_direction_unit
    del selected_ablation_layers, selected_inlp_layers
    del baseline_fwd_pre_hooks, baseline_fwd_hooks
    del ablation_fwd_pre_hooks, ablation_fwd_hooks
    del nullspace_fwd_pre_hooks, nullspace_fwd_hooks

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_pipeline(model_path, device='auto', vllm_gpu_memory_utilization=0.9,
                 resume_from_eval=False, skip_eval=False, use_existing=False,
                 top_percentage=1.0, extract_only=False, select_only=False,
                 infer_only=False, compare_rankings=False):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, device=device,
                 vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                 top_percentage=top_percentage,
                 compare_rankings=compare_rankings)

    save_run_params(cfg, extra_flags={
        'resume_from_eval': resume_from_eval,
        'skip_eval': skip_eval,
        'use_existing': use_existing,
        'extract_only': extract_only,
        'select_only': select_only,
        'infer_only': infer_only,
        'compare_rankings': compare_rankings,
    })

    phase_flags = [extract_only, select_only, infer_only]
    if sum(1 for flag in phase_flags if flag) > 1:
        raise ValueError("Use at most one of --extract_only, --select_only, or --infer_only.")

    if extract_only:
        if resume_from_eval or use_existing:
            raise ValueError("--extract_only is incompatible with --resume_from_eval and --use_existing.")
        _run_extraction(cfg, model_path)
        return

    if select_only:
        if resume_from_eval or use_existing:
            raise ValueError("--select_only is incompatible with --resume_from_eval and --use_existing.")
        _run_selection(cfg, model_path)
        return

    if infer_only:
        _run_inference_from_existing(cfg, model_path)
        if not skip_eval:
            _run_evaluation(cfg)
        return

    if use_existing:
        _run_inference_from_existing(cfg, model_path)
        if not skip_eval:
            _run_evaluation(cfg)
        return

    if not resume_from_eval:
        _run_inference(cfg, model_path)

    if not skip_eval:
        _run_evaluation(cfg)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, device=args.device,
                 vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                 resume_from_eval=args.resume_from_eval,
                 skip_eval=args.skip_eval,
                 use_existing=args.use_existing,
                 top_percentage=args.top_percentage,
                 extract_only=args.extract_only,
                 select_only=args.select_only,
                 infer_only=args.infer_only,
                 compare_rankings=args.compare_rankings)
