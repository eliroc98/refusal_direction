import gc
import glob
import shutil
import torch
import random
import json
import os
import argparse
import math

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config, COMPONENT_MODES, K_POLICIES
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
    get_direction_ablation_hooks,
    get_all_direction_ablation_hooks,
    get_counterfactual_reflection_hooks,
    get_all_counterfactual_reflection_hooks,
)

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.generate_directions_inlp import (
    generate_directions_inlp,
    select_direction_inlp_ranked,
)
from pipeline.submodules.select_direction import select_direction_ranked, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.submodules.evaluate_benchmarks import evaluate_benchmarks

ACTADD_TARGET_MULTIPLIERS = [0.5, 1.0, 2.0]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run refusal-direction intervention pipeline.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--device', type=str, default='auto',
                        help='"auto" to spread across GPUs, or e.g. "cuda:0" / "cpu". (default: auto)')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.9,
                        help='Fraction of GPU memory for vLLM classifiers. (default: 0.9)')
    parser.add_argument('--component_mode', type=str, default='just_1', choices=COMPONENT_MODES,
                        help='"just_1": hook only the best (pos, layer). '
                             '"all": hook every model layer with the single best direction/P '
                             '(andyrdt-style). (default: just_1)')
    parser.add_argument('--inlp_k_policy', type=str, default='none', choices=K_POLICIES,
                        help='"none" uses the full INLP P. "fixed" restricts to --inlp_k_restrict '
                             'directions. "acc90"/"acc80" restrict per (pos, layer) to the number '
                             'of classifiers with dev accuracy >= 0.90 / 0.80. (default: none)')
    parser.add_argument('--inlp_k_restrict', type=int, default=None,
                        help='Integer k for --inlp_k_policy=fixed.')
    parser.add_argument('--reflection_alphas', type=float, nargs='+', default=None,
                        help='Reflection alphas; default (1.0, 2.0).')
    parser.add_argument('--extract_only', action='store_true',
                        help='Run only extraction (datasets, mean-diff, INLP activations).')
    parser.add_argument('--select_only', action='store_true',
                        help='Rank + select best components from pre-extracted artifacts.')
    parser.add_argument('--infer_only', action='store_true',
                        help='Run completions + loss + benchmark evals from pre-selected best components.')
    parser.add_argument('--resume_from_eval', action='store_true',
                        help='Skip inference and run LlamaGuard evaluation on existing completions.')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Run inference; skip LlamaGuard evaluation.')
    parser.add_argument('--force_overwrite', action='store_true',
                        help='Regenerate completion/eval files even if they exist.')
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset sampling & persistence (harmless_test frozen for eval consistency)
# ──────────────────────────────────────────────────────────────────────────────

def load_and_sample_datasets(cfg):
    random.seed(42)
    harmful_train = random.sample(
        load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    harmless_train = random.sample(
        load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    harmful_val = random.sample(
        load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    harmless_val = random.sample(
        load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)
    return harmful_train, harmless_train, harmful_val, harmless_val, harmless_test


def save_dataset_artifacts(cfg, harmful_train, harmless_train, harmful_val, harmless_val, harmless_test):
    payload = {
        "seed": 42,
        "harmful_train": harmful_train,
        "harmless_train": harmless_train,
        "harmful_val": harmful_val,
        "harmless_val": harmless_val,
        "harmless_test": harmless_test,
        "counts": {
            "harmful_train": len(harmful_train),
            "harmless_train": len(harmless_train),
            "harmful_val": len(harmful_val),
            "harmless_val": len(harmless_val),
            "harmless_test": len(harmless_test),
        },
    }
    os.makedirs(cfg.extraction_path(), exist_ok=True)
    with open(os.path.join(cfg.extraction_path(), 'dataset_artifacts.json'), 'w') as f:
        json.dump(payload, f, indent=2)


def load_dataset_artifacts(cfg):
    path = os.path.join(cfg.extraction_path(), 'dataset_artifacts.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing dataset artifacts at {path}. Run --extract_only first."
        )
    with open(path, 'r') as f:
        payload = json.load(f)
    return (
        payload["harmful_train"],
        payload["harmless_train"],
        payload["harmful_val"],
        payload["harmless_val"],
        payload.get("harmless_test"),
    )


def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    def _filter(ds, scores, threshold, cmp):
        return [inst for inst, s in zip(ds, scores.tolist()) if cmp(s, threshold)]

    if cfg.filter_train:
        hf_s = get_refusal_scores(model_base.model, harmful_train,
                                  model_base.tokenize_instructions_fn, model_base.refusal_toks)
        hl_s = get_refusal_scores(model_base.model, harmless_train,
                                  model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = _filter(harmful_train, hf_s, 0, lambda x, y: x > y)
        harmless_train = _filter(harmless_train, hl_s, 0, lambda x, y: x < y)

    if cfg.filter_val:
        hf_vs = get_refusal_scores(model_base.model, harmful_val,
                                   model_base.tokenize_instructions_fn, model_base.refusal_toks)
        hl_vs = get_refusal_scores(model_base.model, harmless_val,
                                   model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = _filter(harmful_val, hf_vs, 0, lambda x, y: x > y)
        harmless_val = _filter(harmless_val, hl_vs, 0, lambda x, y: x < y)

    return harmful_train, harmless_train, harmful_val, harmless_val


# ──────────────────────────────────────────────────────────────────────────────
#  Direction extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    out_dir = os.path.join(cfg.extraction_path(), 'generate_directions')
    os.makedirs(out_dir, exist_ok=True)
    mean_diffs = generate_directions(model_base, harmful_train, harmless_train)
    torch.save(mean_diffs, os.path.join(out_dir, 'mean_diffs.pt'))
    return mean_diffs


def generate_and_save_inlp_directions(cfg, model_base, harmful_train, harmless_train):
    artifact_dir = os.path.join(cfg.extraction_path(), 'generate_directions_inlp')
    generate_directions_inlp(model_base, harmful_train, harmless_train, artifact_dir=artifact_dir)


# ──────────────────────────────────────────────────────────────────────────────
#  Ranking / best-component selection (andyrdt filter)
# ──────────────────────────────────────────────────────────────────────────────

def _make_ablation_component(row, candidate_directions):
    pos = int(row['position']); layer = int(row['layer'])
    comp = {
        'position': pos,
        'layer': layer,
        'refusal_score_local': float(row['refusal_score_local']),
        'steering_median_score_local': float(row['steering_median_score_local']),
        'kl_div_score_local': float(row['kl_div_score_local']),
        'sorting_score_local': float(row['sorting_score_local']),
        'direction': candidate_directions[pos, layer].detach().cpu().float(),
    }
    for key in ('refusal_score', 'steering_median_score', 'kl_div_score', 'sorting_score'):
        comp[key] = float(row.get(key, float('nan')))
    return comp


def _make_inlp_component(row):
    comp = {
        'position': int(row['position']),
        'layer': int(row['layer']),
        'refusal_score': float(row['refusal_score']),
        'steering_score': float(row['steering_score']),
        'kl_div_score': float(row['kl_div_score']),
        'sorting_score': float(row['sorting_score']),
        'direction': torch.from_numpy(row['first_dir']).float(),
        'P': row['P'],
    }
    if 'k_used' in row:
        comp['k_used'] = int(row['k_used'])
    if 'k_accs_used' in row:
        comp['k_accs_used'] = list(row['k_accs_used'])
    return comp


def _safe_float(val, default=None):
    v = float(val)
    return default if math.isnan(v) else v


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
        compare_rankings=False,
    )
    all_components = [_make_ablation_component(row, candidate_directions) for row in all_ranked]
    filtered_components = [_make_ablation_component(row, candidate_directions) for row in filtered_ranked]
    return all_components, filtered_components, top_direction_norm


def select_ranked_inlp_components(cfg, model_base, harmful_val, harmless_val, actadd_multipliers, direction_norm):
    artifact_dir = os.path.join(cfg.extraction_path(), 'generate_directions_inlp')
    all_ranked, filtered_ranked = select_direction_inlp_ranked(
        artifact_dir=artifact_dir,
        model_base=model_base,
        harmful_instructions=harmful_val,
        harmless_instructions=harmless_val,
        actadd_multipliers=actadd_multipliers,
        direction_norm=direction_norm,
        k_policy=cfg.inlp_k_policy,
        k_fixed=cfg.inlp_k_restrict,
    )
    all_components = [_make_inlp_component(row) for row in all_ranked]
    filtered_components = [_make_inlp_component(row) for row in filtered_ranked]
    return all_components, filtered_components


def save_cell_metadata(dir_path, cfg, best_ablation, best_inlp, filtered_ablation_size, filtered_inlp_size):
    os.makedirs(dir_path, exist_ok=True)
    payload = {
        'component_mode': cfg.component_mode,
        'inlp_k_policy': cfg.inlp_k_policy,
        'inlp_k_restrict': cfg.inlp_k_restrict,
        'filtered_ablation_pool_size': filtered_ablation_size,
        'filtered_inlp_pool_size': filtered_inlp_size,
        'best_ablation': {
            'position': best_ablation['position'],
            'layer': best_ablation['layer'],
            'refusal_score_local': _safe_float(best_ablation.get('refusal_score_local', float('nan'))),
        },
        'best_inlp': {
            'position': best_inlp['position'],
            'layer': best_inlp['layer'],
            'k_used': best_inlp.get('k_used'),
            'k_accs_used': best_inlp.get('k_accs_used'),
            'refusal_score': _safe_float(best_inlp.get('refusal_score', float('nan'))),
        },
    }
    with open(os.path.join(dir_path, 'selected_components_metadata.json'), 'w') as f:
        json.dump(payload, f, indent=4)


def save_run_params(dir_path, cfg, extra_flags=None):
    import dataclasses
    os.makedirs(dir_path, exist_ok=True)
    params = dataclasses.asdict(cfg)
    if extra_flags:
        params.update(extra_flags)
    with open(os.path.join(dir_path, 'run_params.json'), 'w') as f:
        json.dump(params, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
#  Completion + eval helpers (dir-scoped so we can write to mode dir or actadd dir)
# ──────────────────────────────────────────────────────────────────────────────

def _try_link_from_siblings(target_path, candidate_sources):
    """Hardlink target_path from the first existing candidate source.
    Returns True if a link/copy was made (skip recompute), False otherwise."""
    for src in candidate_sources:
        if src == target_path or not os.path.exists(src):
            continue
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        try:
            os.link(src, target_path)
        except OSError:
            shutil.copyfile(src, target_path)
        print(f"Reused {target_path} from sibling {src}")
        return True
    return False


def _best_inlp_matches(src_cell_dir, cur_cell_dir):
    """For inlp_actadd reuse: confirm two cells share the best_inlp (pos, layer)."""
    src_meta = os.path.join(src_cell_dir, 'selected_components_metadata.json')
    cur_meta = os.path.join(cur_cell_dir, 'selected_components_metadata.json')
    if not (os.path.exists(src_meta) and os.path.exists(cur_meta)):
        return False
    with open(src_meta) as f:
        s = json.load(f).get('best_inlp', {})
    with open(cur_meta) as f:
        c = json.load(f).get('best_inlp', {})
    return (s.get('position') == c.get('position')
            and s.get('layer') == c.get('layer'))


def _sibling_sources(cfg, scope, subdir, fname):
    """Resolve sibling artifact paths under runs/<alias>/.

    scope ∈ {'any', 'same_mode', 'actadd', 'inlp_actadd'}
    subdir ∈ {'completions', 'loss_evals', 'benchmark_evals'}
    """
    root = cfg.extraction_path()
    if scope == 'any':
        glob_pat = os.path.join(root, '*', subdir, fname)
    elif scope == 'same_mode':
        glob_pat = os.path.join(root, f"{cfg.component_mode}__*", subdir, fname)
    elif scope in ('actadd', 'inlp_actadd'):
        glob_pat = os.path.join(root, 'actadd__*', subdir, fname)
    else:
        return []
    candidates = sorted(glob.glob(glob_pat), key=os.path.getmtime, reverse=True)
    if scope == 'inlp_actadd':
        cur_cell = cfg.actadd_path()
        candidates = [c for c in candidates
                      if _best_inlp_matches(os.path.dirname(os.path.dirname(c)), cur_cell)]
    return candidates


def _maybe_write_completions(cfg, dir_path, model_base, fwd_pre_hooks, fwd_hooks,
                             label, dataset_name, dataset=None, reuse_from=None):
    os.makedirs(os.path.join(dir_path, 'completions'), exist_ok=True)
    out_path = os.path.join(dir_path, 'completions', f'{dataset_name}_{label}_completions.json')
    if not cfg.force_overwrite and os.path.exists(out_path):
        print(f"Skipping {dataset_name}/{label}: {out_path} already exists")
        return
    if not cfg.force_overwrite and reuse_from and _try_link_from_siblings(out_path, reuse_from):
        return
    if dataset is None:
        dataset = load_dataset(dataset_name)
    completions = model_base.generate_completions(
        dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks,
        max_new_tokens=cfg.max_new_tokens,
    )
    with open(out_path, 'w') as f:
        json.dump(completions, f, indent=4)


def _maybe_evaluate_completions(cfg, dir_path, label, dataset_name, methodologies,
                                lg2_classifier=None, reuse_from=None):
    comp_path = os.path.join(dir_path, 'completions', f'{dataset_name}_{label}_completions.json')
    eval_path = os.path.join(dir_path, 'completions', f'{dataset_name}_{label}_evaluations.json')
    if not os.path.exists(comp_path):
        print(f"Skipping eval {dataset_name}/{label}: completions file missing")
        return
    if not cfg.force_overwrite and os.path.exists(eval_path):
        print(f"Skipping eval {dataset_name}/{label}: {eval_path} already exists")
        return
    if not cfg.force_overwrite and reuse_from and _try_link_from_siblings(eval_path, reuse_from):
        return
    with open(comp_path, 'r') as f:
        completions = json.load(f)
    evaluate_jailbreak(
        completions=completions,
        methodologies=methodologies,
        evaluation_path=eval_path,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        llamaguard2_classifier=lg2_classifier,
    )


def _maybe_evaluate_loss(cfg, dir_path, model_base, fwd_pre_hooks, fwd_hooks, label,
                         reuse_from=None):
    os.makedirs(os.path.join(dir_path, 'loss_evals'), exist_ok=True)
    out_path = os.path.join(dir_path, 'loss_evals', f'{label}_loss_eval.json')
    if not cfg.force_overwrite and os.path.exists(out_path):
        print(f"Skipping loss eval {label}: already exists")
        return
    if not cfg.force_overwrite and reuse_from and _try_link_from_siblings(out_path, reuse_from):
        return
    on_dist_path = os.path.join(dir_path, 'completions', 'harmless_baseline_completions.json')
    loss_evals = evaluate_loss(
        model_base, fwd_pre_hooks, fwd_hooks,
        batch_size=cfg.ce_loss_batch_size,
        n_batches=cfg.ce_loss_n_batches,
        completions_file_path=on_dist_path,
        intervention_label=label,
    )
    with open(out_path, 'w') as f:
        json.dump(loss_evals, f, indent=4)


def _maybe_evaluate_benchmarks(cfg, dir_path, model_base, fwd_pre_hooks, fwd_hooks, label,
                               reuse_from=None):
    os.makedirs(os.path.join(dir_path, 'benchmark_evals'), exist_ok=True)
    out_path = os.path.join(dir_path, 'benchmark_evals', f'{label}_benchmark_eval.json')
    if not cfg.force_overwrite and os.path.exists(out_path):
        print(f"Skipping benchmark eval {label}: already exists")
        return
    if not cfg.force_overwrite and reuse_from and _try_link_from_siblings(out_path, reuse_from):
        return
    evals = evaluate_benchmarks(
        model_base, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks,
        n_mmlu=cfg.benchmark_n_mmlu, n_arc=cfg.benchmark_n_arc,
        n_truthfulqa=0,
        intervention_label=label,
    )
    with open(out_path, 'w') as f:
        json.dump(evals, f, indent=4)


# ──────────────────────────────────────────────────────────────────────────────
#  Mode-dependent hook construction (ablation + reflection)
# ──────────────────────────────────────────────────────────────────────────────

def _build_ablation_hooks(cfg, model_base, best_layer, best_direction):
    if cfg.component_mode == 'just_1':
        return get_direction_ablation_hooks(
            model_base, [{'layer': best_layer}], best_direction)
    return get_all_direction_ablation_hooks(model_base, best_direction)


def _build_reflection_hooks(cfg, model_base, best_inlp_layer, best_P, alpha):
    if cfg.component_mode == 'just_1':
        return get_counterfactual_reflection_hooks(
            model_base, [{'layer': best_inlp_layer, 'P': best_P}], alpha=alpha)
    return get_all_counterfactual_reflection_hooks(model_base, best_P, alpha=alpha)


# ──────────────────────────────────────────────────────────────────────────────
#  Pipeline phases
# ──────────────────────────────────────────────────────────────────────────────

def _run_extraction(cfg):
    """Sample & freeze datasets, compute mean-diff and INLP artifacts. One-time per model."""
    model_base = construct_model_base(cfg.model_path, device=cfg.device)

    h_train, hl_train, h_val, hl_val, hl_test = load_and_sample_datasets(cfg)
    h_train, hl_train, h_val, hl_val = filter_data(cfg, model_base, h_train, hl_train, h_val, hl_val)

    save_dataset_artifacts(cfg, h_train, hl_train, h_val, hl_val, hl_test)
    generate_and_save_candidate_directions(cfg, model_base, h_train, hl_train)
    generate_and_save_inlp_directions(cfg, model_base, h_train, hl_train)

    model_base.del_model()
    del model_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _selection_cache_path(cell_dir):
    return os.path.join(cell_dir, 'selected_components.pt')


def _sibling_selection_caches(cfg):
    """Sibling selection caches for the same k (any mode); selection is
    mode-independent, so just_1__<k> and all__<k> share the same result."""
    root = cfg.extraction_path()
    pat = os.path.join(root, f'*__{cfg.k_label()}', 'selected_components.pt')
    return sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)


def _select_best_components(cfg, model_base):
    """Rank candidates and return (best_ablation, best_inlp, n_ablation, n_inlp, top_direction_norm).
    Cached per-cell at <cell>/selected_components.pt; reused across modes for the same k."""
    cache_path = _selection_cache_path(cfg.artifact_path())

    if not cfg.force_overwrite:
        if not os.path.exists(cache_path):
            _try_link_from_siblings(cache_path, _sibling_selection_caches(cfg))
        if os.path.exists(cache_path):
            print(f"Loading cached selection from {cache_path}")
            cached = torch.load(cache_path, map_location='cpu', weights_only=False)
            return (cached['best_ablation'], cached['best_inlp'],
                    cached['n_abl'], cached['n_inlp'], cached['top_direction_norm'])

    h_train, hl_train, h_val, hl_val, _ = load_dataset_artifacts(cfg)

    mean_diffs_path = os.path.join(cfg.extraction_path(), 'generate_directions', 'mean_diffs.pt')
    if not os.path.exists(mean_diffs_path):
        raise FileNotFoundError(f"Missing {mean_diffs_path}. Run --extract_only first.")
    candidate_directions = torch.load(mean_diffs_path, map_location='cpu', weights_only=True)

    actadd_multipliers = ACTADD_TARGET_MULTIPLIERS
    _, filtered_ablation, top_direction_norm = select_ranked_direction_components(
        cfg, model_base, h_val, hl_val, candidate_directions, actadd_multipliers)
    _, filtered_inlp = select_ranked_inlp_components(
        cfg, model_base, h_val, hl_val, actadd_multipliers, top_direction_norm)

    if len(filtered_ablation) == 0:
        raise RuntimeError("No mean-diff components survived filtering — cannot proceed.")
    if len(filtered_inlp) == 0:
        raise RuntimeError("No INLP components survived filtering under current k-policy — cannot proceed.")

    best_ablation = filtered_ablation[0]
    best_inlp = filtered_inlp[0]
    n_abl, n_inlp = len(filtered_ablation), len(filtered_inlp)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({
        'best_ablation': best_ablation,
        'best_inlp': best_inlp,
        'n_abl': n_abl,
        'n_inlp': n_inlp,
        'top_direction_norm': top_direction_norm,
    }, cache_path)

    return best_ablation, best_inlp, n_abl, n_inlp, top_direction_norm


def _run_selection(cfg):
    """Rank + write metadata for the current (component_mode, k) cell."""
    model_base = construct_model_base(cfg.model_path, device=cfg.device)
    best_ablation, best_inlp, n_abl, n_inlp, _ = _select_best_components(cfg, model_base)
    save_cell_metadata(cfg.artifact_path(), cfg, best_ablation, best_inlp, n_abl, n_inlp)
    save_cell_metadata(cfg.actadd_path(), cfg, best_ablation, best_inlp, n_abl, n_inlp)
    print(
        f"Selection done: mode={cfg.component_mode}, k={cfg.k_label()}, "
        f"best_ablation=(pos={best_ablation['position']}, layer={best_ablation['layer']}), "
        f"best_inlp=(pos={best_inlp['position']}, layer={best_inlp['layer']}, "
        f"k_used={best_inlp.get('k_used')})"
    )
    model_base.del_model()
    del model_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_inference(cfg):
    """Run all completions + loss + benchmark evals for this (mode, k) cell.

    Writes mode-specific results (ablation + reflection α=1 + α=2) to cfg.artifact_path()
    and shared-across-modes actadd results to cfg.actadd_path().
    """
    model_base = construct_model_base(cfg.model_path, device=cfg.device)
    best_ablation, best_inlp, n_abl, n_inlp, _ = _select_best_components(cfg, model_base)

    best_direction = best_ablation['direction']
    best_layer = best_ablation['layer']
    best_inlp_direction = best_inlp['direction']
    best_inlp_layer = best_inlp['layer']

    # best_inlp['P'] is already the k-restricted projection built from the
    # classifier-index subset (or full P for policy='none') by
    # select_direction_inlp_ranked — no further restriction here.
    best_P = best_inlp['P']

    direction_norm = torch.norm(best_direction).item()
    direction_unit = best_direction / (direction_norm + 1e-8)
    inlp_direction_unit = best_inlp_direction / (torch.norm(best_inlp_direction) + 1e-8)
    actadd_coeffs = [m * direction_norm for m in ACTADD_TARGET_MULTIPLIERS]

    mode_dir = cfg.artifact_path()
    actadd_dir = cfg.actadd_path()

    save_cell_metadata(mode_dir, cfg, best_ablation, best_inlp, n_abl, n_inlp)
    save_cell_metadata(actadd_dir, cfg, best_ablation, best_inlp, n_abl, n_inlp)
    save_run_params(mode_dir, cfg)
    save_run_params(actadd_dir, cfg)

    # Persist actadd coefficients so evaluation can discover the sweep.
    with open(os.path.join(actadd_dir, 'actadd_coeffs.json'), 'w') as f:
        json.dump({
            "direction_norm": direction_norm,
            "multipliers": ACTADD_TARGET_MULTIPLIERS,
            "coeffs": actadd_coeffs,
            "reflection_alphas": list(cfg.reflection_alphas),
        }, f, indent=2)

    print(
        f"Inference cell: mode={cfg.component_mode}, k={cfg.k_label()}, "
        f"direction_norm={direction_norm:.4f}, "
        f"actadd_coeffs={[f'{c:.2f}' for c in actadd_coeffs]}"
    )

    # Load frozen harmless_test once (falls back to fresh sampling if extraction predates the freeze).
    _, _, _, _, harmless_test = load_dataset_artifacts(cfg)
    if harmless_test is None:
        random.seed(42)
        harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    baseline_pre, baseline_post = [], []
    ablation_pre, ablation_post = _build_ablation_hooks(cfg, model_base, best_layer, best_direction)

    # ── Mode dir: baseline + ablation + reflection ──────────────────────────
    for dataset_name in cfg.evaluation_datasets:
        _maybe_write_completions(cfg, mode_dir, model_base, baseline_pre, baseline_post,
                                 'baseline', dataset_name,
                                 reuse_from=_sibling_sources(cfg, 'any', 'completions',
                                     f'{dataset_name}_baseline_completions.json'))
        _maybe_write_completions(cfg, mode_dir, model_base, ablation_pre, ablation_post,
                                 'ablation', dataset_name,
                                 reuse_from=_sibling_sources(cfg, 'same_mode', 'completions',
                                     f'{dataset_name}_ablation_completions.json'))
        for alpha in cfg.reflection_alphas:
            label = f'reflection_a{alpha:.2f}'
            refl_pre, refl_post = _build_reflection_hooks(cfg, model_base, best_inlp_layer, best_P, alpha)
            _maybe_write_completions(cfg, mode_dir, model_base, refl_pre, refl_post,
                                     label, dataset_name)

    _maybe_write_completions(cfg, mode_dir, model_base, baseline_pre, baseline_post,
                             'baseline', 'harmless', dataset=harmless_test,
                             reuse_from=_sibling_sources(cfg, 'any', 'completions',
                                 'harmless_baseline_completions.json'))
    for alpha in cfg.reflection_alphas:
        label = f'reflection_a{alpha:.2f}'
        refl_pre, refl_post = _build_reflection_hooks(cfg, model_base, best_inlp_layer, best_P, alpha)
        _maybe_write_completions(cfg, mode_dir, model_base, refl_pre, refl_post,
                                 label, 'harmless', dataset=harmless_test)

    # Loss + benchmark evals for mode dir
    _maybe_evaluate_loss(cfg, mode_dir, model_base, baseline_pre, baseline_post, 'baseline',
                         reuse_from=_sibling_sources(cfg, 'any', 'loss_evals', 'baseline_loss_eval.json'))
    _maybe_evaluate_benchmarks(cfg, mode_dir, model_base, baseline_pre, baseline_post, 'baseline',
                               reuse_from=_sibling_sources(cfg, 'any', 'benchmark_evals', 'baseline_benchmark_eval.json'))
    _maybe_evaluate_loss(cfg, mode_dir, model_base, ablation_pre, ablation_post, 'ablation',
                         reuse_from=_sibling_sources(cfg, 'same_mode', 'loss_evals', 'ablation_loss_eval.json'))
    _maybe_evaluate_benchmarks(cfg, mode_dir, model_base, ablation_pre, ablation_post, 'ablation',
                               reuse_from=_sibling_sources(cfg, 'same_mode', 'benchmark_evals', 'ablation_benchmark_eval.json'))
    for alpha in cfg.reflection_alphas:
        label = f'reflection_a{alpha:.2f}'
        refl_pre, refl_post = _build_reflection_hooks(cfg, model_base, best_inlp_layer, best_P, alpha)
        _maybe_evaluate_loss(cfg, mode_dir, model_base, refl_pre, refl_post, label)
        _maybe_evaluate_benchmarks(cfg, mode_dir, model_base, refl_pre, refl_post, label)

    # ── Actadd dir: baseline + actadd + inlp_actadd (shared across modes) ───
    for dataset_name in cfg.evaluation_datasets:
        _maybe_write_completions(cfg, actadd_dir, model_base, baseline_pre, baseline_post,
                                 'baseline', dataset_name,
                                 reuse_from=_sibling_sources(cfg, 'any', 'completions',
                                     f'{dataset_name}_baseline_completions.json'))
        for coeff in actadd_coeffs:
            aa_label = f'actadd_c{coeff:.2f}'
            aa_hooks = [(model_base.model_block_modules[best_layer],
                         get_activation_addition_input_pre_hook(vector=direction_unit, coeff=-coeff))]
            _maybe_write_completions(cfg, actadd_dir, model_base, aa_hooks, [], aa_label, dataset_name,
                                     reuse_from=_sibling_sources(cfg, 'actadd', 'completions',
                                         f'{dataset_name}_{aa_label}_completions.json'))

            inlp_label = f'inlp_actadd_c{coeff:.2f}'
            inlp_hooks = [(model_base.model_block_modules[best_inlp_layer],
                           get_activation_addition_input_pre_hook(vector=inlp_direction_unit, coeff=-coeff))]
            _maybe_write_completions(cfg, actadd_dir, model_base, inlp_hooks, [], inlp_label, dataset_name,
                                     reuse_from=_sibling_sources(cfg, 'inlp_actadd', 'completions',
                                         f'{dataset_name}_{inlp_label}_completions.json'))

    _maybe_write_completions(cfg, actadd_dir, model_base, baseline_pre, baseline_post,
                             'baseline', 'harmless', dataset=harmless_test,
                             reuse_from=_sibling_sources(cfg, 'any', 'completions',
                                 'harmless_baseline_completions.json'))
    for coeff in actadd_coeffs:
        aa_label = f'actadd_c{coeff:.2f}'
        aa_hooks = [(model_base.model_block_modules[best_layer],
                     get_activation_addition_input_pre_hook(vector=direction_unit, coeff=+coeff))]
        _maybe_write_completions(cfg, actadd_dir, model_base, aa_hooks, [], aa_label,
                                 'harmless', dataset=harmless_test,
                                 reuse_from=_sibling_sources(cfg, 'actadd', 'completions',
                                     f'harmless_{aa_label}_completions.json'))

        inlp_label = f'inlp_actadd_c{coeff:.2f}'
        inlp_hooks = [(model_base.model_block_modules[best_inlp_layer],
                       get_activation_addition_input_pre_hook(vector=inlp_direction_unit, coeff=+coeff))]
        _maybe_write_completions(cfg, actadd_dir, model_base, inlp_hooks, [], inlp_label,
                                 'harmless', dataset=harmless_test,
                                 reuse_from=_sibling_sources(cfg, 'inlp_actadd', 'completions',
                                     f'harmless_{inlp_label}_completions.json'))

    # Loss + benchmark evals for actadd dir
    _maybe_evaluate_loss(cfg, actadd_dir, model_base, baseline_pre, baseline_post, 'baseline',
                         reuse_from=_sibling_sources(cfg, 'any', 'loss_evals', 'baseline_loss_eval.json'))
    _maybe_evaluate_benchmarks(cfg, actadd_dir, model_base, baseline_pre, baseline_post, 'baseline',
                               reuse_from=_sibling_sources(cfg, 'any', 'benchmark_evals', 'baseline_benchmark_eval.json'))
    for coeff in actadd_coeffs:
        aa_label = f'actadd_c{coeff:.2f}'
        aa_hooks = [(model_base.model_block_modules[best_layer],
                     get_activation_addition_input_pre_hook(vector=direction_unit, coeff=-coeff))]
        _maybe_evaluate_loss(cfg, actadd_dir, model_base, aa_hooks, [], aa_label,
                             reuse_from=_sibling_sources(cfg, 'actadd', 'loss_evals', f'{aa_label}_loss_eval.json'))
        _maybe_evaluate_benchmarks(cfg, actadd_dir, model_base, aa_hooks, [], aa_label,
                                   reuse_from=_sibling_sources(cfg, 'actadd', 'benchmark_evals', f'{aa_label}_benchmark_eval.json'))

        inlp_label = f'inlp_actadd_c{coeff:.2f}'
        inlp_hooks = [(model_base.model_block_modules[best_inlp_layer],
                       get_activation_addition_input_pre_hook(vector=inlp_direction_unit, coeff=-coeff))]
        _maybe_evaluate_loss(cfg, actadd_dir, model_base, inlp_hooks, [], inlp_label,
                             reuse_from=_sibling_sources(cfg, 'inlp_actadd', 'loss_evals', f'{inlp_label}_loss_eval.json'))
        _maybe_evaluate_benchmarks(cfg, actadd_dir, model_base, inlp_hooks, [], inlp_label,
                                   reuse_from=_sibling_sources(cfg, 'inlp_actadd', 'benchmark_evals', f'{inlp_label}_benchmark_eval.json'))

    model_base.del_model()
    del model_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_evaluation(cfg):
    """LlamaGuard2 / substring evaluation on completions in mode dir + actadd dir."""
    if cfg.device not in ('auto', 'cpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device.split(':')[-1]

    from pipeline.submodules.evaluate_jailbreak import LlamaGuard2Classifier
    lg2 = None
    if "llamaguard2" in cfg.jailbreak_eval_methodologies:
        lg2 = LlamaGuard2Classifier(gpu_memory_utilization=cfg.vllm_gpu_memory_utilization)

    # Mode dir: baseline + ablation + reflection labels
    mode_dir = cfg.artifact_path()
    if os.path.isdir(mode_dir):
        for dataset_name in cfg.evaluation_datasets:
            _maybe_evaluate_completions(cfg, mode_dir, 'baseline', dataset_name,
                                        cfg.jailbreak_eval_methodologies, lg2,
                                        reuse_from=_sibling_sources(cfg, 'any', 'completions',
                                            f'{dataset_name}_baseline_evaluations.json'))
            _maybe_evaluate_completions(cfg, mode_dir, 'ablation', dataset_name,
                                        cfg.jailbreak_eval_methodologies, lg2,
                                        reuse_from=_sibling_sources(cfg, 'same_mode', 'completions',
                                            f'{dataset_name}_ablation_evaluations.json'))
            for alpha in cfg.reflection_alphas:
                _maybe_evaluate_completions(cfg, mode_dir, f'reflection_a{alpha:.2f}',
                                            dataset_name, cfg.jailbreak_eval_methodologies, lg2)
        _maybe_evaluate_completions(cfg, mode_dir, 'baseline', 'harmless',
                                    cfg.refusal_eval_methodologies,
                                    reuse_from=_sibling_sources(cfg, 'any', 'completions',
                                        'harmless_baseline_evaluations.json'))
        for alpha in cfg.reflection_alphas:
            _maybe_evaluate_completions(cfg, mode_dir, f'reflection_a{alpha:.2f}',
                                        'harmless', cfg.refusal_eval_methodologies)

    # Actadd dir: baseline + actadd + inlp_actadd labels
    actadd_dir = cfg.actadd_path()
    coeffs_path = os.path.join(actadd_dir, 'actadd_coeffs.json')
    actadd_coeffs = []
    if os.path.exists(coeffs_path):
        with open(coeffs_path) as f:
            actadd_coeffs = json.load(f).get('coeffs', [])

    if os.path.isdir(actadd_dir):
        for dataset_name in cfg.evaluation_datasets:
            _maybe_evaluate_completions(cfg, actadd_dir, 'baseline', dataset_name,
                                        cfg.jailbreak_eval_methodologies, lg2,
                                        reuse_from=_sibling_sources(cfg, 'any', 'completions',
                                            f'{dataset_name}_baseline_evaluations.json'))
            for coeff in actadd_coeffs:
                aa_label = f'actadd_c{coeff:.2f}'
                _maybe_evaluate_completions(cfg, actadd_dir, aa_label,
                                            dataset_name, cfg.jailbreak_eval_methodologies, lg2,
                                            reuse_from=_sibling_sources(cfg, 'actadd', 'completions',
                                                f'{dataset_name}_{aa_label}_evaluations.json'))
                inlp_label = f'inlp_actadd_c{coeff:.2f}'
                _maybe_evaluate_completions(cfg, actadd_dir, inlp_label,
                                            dataset_name, cfg.jailbreak_eval_methodologies, lg2,
                                            reuse_from=_sibling_sources(cfg, 'inlp_actadd', 'completions',
                                                f'{dataset_name}_{inlp_label}_evaluations.json'))
        _maybe_evaluate_completions(cfg, actadd_dir, 'baseline', 'harmless',
                                    cfg.refusal_eval_methodologies,
                                    reuse_from=_sibling_sources(cfg, 'any', 'completions',
                                        'harmless_baseline_evaluations.json'))
        for coeff in actadd_coeffs:
            aa_label = f'actadd_c{coeff:.2f}'
            _maybe_evaluate_completions(cfg, actadd_dir, aa_label,
                                        'harmless', cfg.refusal_eval_methodologies,
                                        reuse_from=_sibling_sources(cfg, 'actadd', 'completions',
                                            f'harmless_{aa_label}_evaluations.json'))
            inlp_label = f'inlp_actadd_c{coeff:.2f}'
            _maybe_evaluate_completions(cfg, actadd_dir, inlp_label,
                                        'harmless', cfg.refusal_eval_methodologies,
                                        reuse_from=_sibling_sources(cfg, 'inlp_actadd', 'completions',
                                            f'harmless_{inlp_label}_evaluations.json'))

    if lg2 is not None:
        lg2.cleanup()


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(model_path, device='auto', vllm_gpu_memory_utilization=0.9,
                 component_mode='just_1', inlp_k_policy='none', inlp_k_restrict=None,
                 reflection_alphas=None, extract_only=False, select_only=False,
                 infer_only=False, resume_from_eval=False, skip_eval=False,
                 force_overwrite=False):
    model_alias = os.path.basename(model_path)
    cfg = Config(
        model_alias=model_alias, model_path=model_path, device=device,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        component_mode=component_mode,
        inlp_k_policy=inlp_k_policy, inlp_k_restrict=inlp_k_restrict,
        reflection_alphas=tuple(reflection_alphas) if reflection_alphas else (1.0, 2.0),
        force_overwrite=force_overwrite,
    )

    phase_flags = [extract_only, select_only, infer_only]
    if sum(1 for f in phase_flags if f) > 1:
        raise ValueError("Use at most one of --extract_only, --select_only, --infer_only.")

    if extract_only:
        if resume_from_eval:
            raise ValueError("--extract_only is incompatible with --resume_from_eval.")
        _run_extraction(cfg)
        return

    if select_only:
        if resume_from_eval:
            raise ValueError("--select_only is incompatible with --resume_from_eval.")
        _run_selection(cfg)
        return

    if not resume_from_eval:
        _run_inference(cfg)

    if not skip_eval:
        _run_evaluation(cfg)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(
        model_path=args.model_path,
        device=args.device,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        component_mode=args.component_mode,
        inlp_k_policy=args.inlp_k_policy,
        inlp_k_restrict=args.inlp_k_restrict,
        reflection_alphas=args.reflection_alphas,
        extract_only=args.extract_only,
        select_only=args.select_only,
        infer_only=args.infer_only,
        resume_from_eval=args.resume_from_eval,
        skip_eval=args.skip_eval,
        force_overwrite=args.force_overwrite,
    )
