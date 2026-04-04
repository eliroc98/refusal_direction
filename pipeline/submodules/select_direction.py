import json
import torch
import functools
import math
import matplotlib.pyplot as plt
import os

from typing import List, Optional
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from einops import rearrange

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.nnsight_interventions import (
    LayerIntervention,
    apply_interventions,
    make_ablation_interventions,
    make_actadd_interventions,
)

def refusal_score(
    logits: Float[Tensor, 'batch seq d_vocab_out'],
    refusal_toks: Int[Tensor, 'batch seq'],
    epsilon: Float = 1e-8,
):
    logits = logits.to(torch.float64)

    # we only care about the last tok position
    logits = logits[:, -1, :]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, refusal_toks].sum(dim=-1)

    nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)

def get_refusal_scores(model_base: ModelBase, instructions, tokenize_instructions_fn, refusal_toks,
                       interventions: Optional[List[LayerIntervention]] = None,
                       batch_size=32):
    """Compute refusal log-odds for each instruction using nnsight vLLM tracing."""
    refusal_score_fn = functools.partial(refusal_score, refusal_toks=refusal_toks)
    refusal_scores = torch.zeros(len(instructions))

    # nnsight 0.6.3 cannot push batch results back to the outer frame,
    # so we run one trace per prompt (same pattern as model_base.py).
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i+batch_size]
        prompts = [model_base.format_instruction_fn(inst) for inst in batch]

        last_logits = []
        for prompt in prompts:
            with torch.no_grad():
                with model_base.nnsight_model.trace(prompt, temperature=0.0, top_p=1):
                    if interventions:
                        apply_interventions(model_base, interventions)
                    logit_save = model_base._get_lm_head_proxy().output.save()

            v = logit_save.value
            if v.dim() == 2:
                last_logits.append(v[-1:, :].cpu())      # (1, vocab)
            else:
                last_logits.append(v[0, -1:, :].cpu())   # (1, vocab)

        batch_logits = torch.stack(last_logits, dim=0)  # (batch, 1, vocab)
        refusal_scores[i:i+len(batch)] = refusal_score_fn(logits=batch_logits)

    return refusal_scores

def get_last_position_logits(model_base: ModelBase, instructions, tokenize_instructions_fn=None,
                              interventions: Optional[List[LayerIntervention]] = None,
                              batch_size=32) -> Float[Tensor, "n_instructions d_vocab"]:
    """Return last-position logits for each instruction using nnsight vLLM tracing."""
    last_position_logits = None

    # nnsight 0.6.3 cannot push batch results back to the outer frame,
    # so we run one trace per prompt (same pattern as model_base.py).
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i+batch_size]
        prompts = [model_base.format_instruction_fn(inst) for inst in batch]

        batch_logits_list = []
        for prompt in prompts:
            with torch.no_grad():
                with model_base.nnsight_model.trace(prompt, temperature=0.0, top_p=1):
                    if interventions:
                        apply_interventions(model_base, interventions)
                    logit_save = model_base._get_lm_head_proxy().output.save()

            v = logit_save.value
            if v.dim() == 2:
                batch_logits_list.append(v[-1, :].cpu())      # (vocab,)
            else:
                batch_logits_list.append(v[0, -1, :].cpu())   # (vocab,)

        batch_logits = torch.stack(batch_logits_list, dim=0)  # (batch, vocab)

        if last_position_logits is None:
            last_position_logits = batch_logits
        else:
            last_position_logits = torch.cat([last_position_logits, batch_logits], dim=0)

    return last_position_logits

def plot_refusal_scores(
    refusal_scores: Float[Tensor, 'n_pos n_layer'],
    baseline_refusal_score: Optional[float],
    token_labels: List[str],
    title: str,
    artifact_dir: str,
    artifact_name: str,
):
    n_pos, n_layer = refusal_scores.shape

    fig, ax = plt.subplots(figsize=(9, 5))

    for i in range(-n_pos, 0):
        ax.plot(
            list(range(n_layer)),
            refusal_scores[i].cpu().numpy(),
            label=f'{i}: {repr(token_labels[i])}'
        )

    if baseline_refusal_score is not None:
        ax.axhline(y=baseline_refusal_score, color='black', linestyle='--')
        ax.annotate('Baseline', xy=(1, baseline_refusal_score), xytext=(8, 10),
                    xycoords=('axes fraction', 'data'), textcoords='offset points',
                    horizontalalignment='right', verticalalignment='center')

    ax.set_title(title)
    ax.set_xlabel('Layer source of direction (resid_pre)')
    ax.set_ylabel('Refusal score')
    ax.legend(title='Position source of direction', loc='lower left')

    plt.savefig(f"{artifact_dir}/{artifact_name}.png")

# returns True if the direction should be filtered out
def filter_fn(refusal_score, steering_score, kl_div_score, layer, n_layer, kl_threshold=None, induce_refusal_threshold=None, prune_layer_percentage=0.20) -> bool:
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if induce_refusal_threshold is not None and steering_score < induce_refusal_threshold:
        return True
    return False


def _nan_safe_sort_key(x):
    """Sort key that pushes NaN sorting_score_local to the bottom of the ranking."""
    score = x['sorting_score_local']
    return (float('inf') if math.isnan(score) else -score, x['position'], x['layer'])


def _sanitize_for_json(obj):
    """Replace NaN float values with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def select_direction_ranked(
    model_base: ModelBase,
    harmful_instructions,
    harmless_instructions,
    candidate_directions: Float[Tensor, 'n_pos n_layer d_model'],
    artifact_dir,
    actadd_multipliers: list,
    kl_threshold=0.1,
    induce_refusal_threshold=0.0,
    prune_layer_percentage=0.2,
    batch_size=32,
    compare_rankings=False,
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_pos, n_layer, d_model = candidate_directions.shape

    baseline_refusal_scores_harmful = get_refusal_scores(
        model_base, harmful_instructions, model_base.tokenize_instructions_fn,
        model_base.refusal_toks, batch_size=batch_size)
    baseline_refusal_scores_harmless = get_refusal_scores(
        model_base, harmless_instructions, model_base.tokenize_instructions_fn,
        model_base.refusal_toks, batch_size=batch_size)

    ablation_kl_div_scores = torch.zeros((n_pos, n_layer), dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_pos, n_layer), dtype=torch.float64)

    ablation_kl_div_scores_local = torch.zeros((n_pos, n_layer), dtype=torch.float64)
    ablation_refusal_scores_local = torch.zeros((n_pos, n_layer), dtype=torch.float64)

    steering_median_scores = torch.zeros((n_pos, n_layer), dtype=torch.float64)
    steering_median_scores_local = torch.zeros((n_pos, n_layer), dtype=torch.float64)

    baseline_harmless_logits = get_last_position_logits(
        model_base=model_base,
        instructions=harmless_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        batch_size=batch_size,
    )

    # --- Global (all-layer) scoring: only when compare_rankings is requested ---
    if compare_rankings:
        for source_pos in range(-n_pos, 0):
            for source_layer in tqdm(range(n_layer), desc=f"Computing KL for source position {source_pos}"):
                ablation_dir = candidate_directions[source_pos, source_layer]
                # All-layer ablation: apply at every layer
                interventions = make_ablation_interventions(list(range(n_layer)), ablation_dir)

                intervention_logits = get_last_position_logits(
                    model_base=model_base,
                    instructions=harmless_instructions,
                    tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                    interventions=interventions,
                    batch_size=batch_size,
                )
                ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(
                    baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

        for source_pos in range(-n_pos, 0):
            for source_layer in tqdm(range(n_layer), desc=f"Computing refusal ablation for source position {source_pos}"):
                ablation_dir = candidate_directions[source_pos, source_layer]
                interventions = make_ablation_interventions(list(range(n_layer)), ablation_dir)

                scores = get_refusal_scores(
                    model_base, harmful_instructions, model_base.tokenize_instructions_fn,
                    model_base.refusal_toks, interventions=interventions, batch_size=batch_size)
                ablation_refusal_scores[source_pos, source_layer] = scores.mean().item()

        # Steering median (all-layer): actadd at every layer on harmless
        for source_pos in range(-n_pos, 0):
            for source_layer in tqdm(range(n_layer), desc=f"Computing steering median (all-layer) for source position {source_pos}"):
                direction = candidate_directions[source_pos, source_layer]
                dir_norm = torch.norm(direction).item()
                direction_unit = direction / (torch.norm(direction) + 1e-8)

                per_mult_scores = []
                for m in actadd_multipliers:
                    coeff = m * dir_norm
                    interventions = []
                    for layer in range(n_layer):
                        interventions.extend(make_actadd_interventions(layer, direction_unit, coeff))

                    scores = get_refusal_scores(
                        model_base, harmless_instructions,
                        model_base.tokenize_instructions_fn, model_base.refusal_toks,
                        interventions=interventions, batch_size=batch_size)
                    per_mult_scores.append(scores.mean().item())

                steering_median_scores[source_pos, source_layer] = float(torch.tensor(per_mult_scores).max())
    else:
        ablation_kl_div_scores.fill_(float('nan'))
        ablation_refusal_scores.fill_(float('nan'))
        steering_median_scores.fill_(float('nan'))

    # Steering median (local): actadd at source layer only on harmless
    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing steering median (local) for source position {source_pos}"):
            direction = candidate_directions[source_pos, source_layer]
            dir_norm = torch.norm(direction).item()
            direction_unit = direction / (torch.norm(direction) + 1e-8)

            per_mult_scores = []
            for m in actadd_multipliers:
                coeff = m * dir_norm
                interventions = make_actadd_interventions(source_layer, direction_unit, coeff)
                scores = get_refusal_scores(
                    model_base, harmless_instructions,
                    model_base.tokenize_instructions_fn, model_base.refusal_toks,
                    interventions=interventions, batch_size=batch_size)
                per_mult_scores.append(scores.mean().item())

            steering_median_scores_local[source_pos, source_layer] = float(torch.tensor(per_mult_scores).max())

    # --- Per-component (local) scoring: ablation at only the source layer ---
    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing LOCAL KL for source position {source_pos}"):
            ablation_dir = candidate_directions[source_pos, source_layer]
            interventions = make_ablation_interventions([source_layer], ablation_dir)

            intervention_logits = get_last_position_logits(
                model_base=model_base,
                instructions=harmless_instructions,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                interventions=interventions,
                batch_size=batch_size,
            )
            ablation_kl_div_scores_local[source_pos, source_layer] = kl_div_fn(
                baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing LOCAL refusal ablation for source position {source_pos}"):
            ablation_dir = candidate_directions[source_pos, source_layer]
            interventions = make_ablation_interventions([source_layer], ablation_dir)

            scores = get_refusal_scores(
                model_base, harmful_instructions, model_base.tokenize_instructions_fn,
                model_base.refusal_toks, interventions=interventions, batch_size=batch_size)
            ablation_refusal_scores_local[source_pos, source_layer] = scores.mean().item()

    # --- Plots ---
    token_labels = model_base.tokenizer.batch_decode(model_base.eoi_toks)

    if compare_rankings:
        plot_refusal_scores(
            refusal_scores=ablation_refusal_scores,
            baseline_refusal_score=baseline_refusal_scores_harmful.mean().item(),
            token_labels=token_labels,
            title='Ablating direction on harmful instructions',
            artifact_dir=artifact_dir,
            artifact_name='ablation_scores'
        )
        plot_refusal_scores(
            refusal_scores=ablation_kl_div_scores,
            baseline_refusal_score=0.0,
            token_labels=token_labels,
            title='KL Divergence when ablating direction on harmless instructions',
            artifact_dir=artifact_dir,
            artifact_name='kl_div_scores'
        )
        plot_refusal_scores(
            refusal_scores=steering_median_scores,
            baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
            token_labels=token_labels,
            title=f'Steering median (all-layer actadd on harmless, multipliers={actadd_multipliers})',
            artifact_dir=artifact_dir,
            artifact_name='steering_median_scores'
        )

    plot_refusal_scores(
        refusal_scores=ablation_refusal_scores_local,
        baseline_refusal_score=baseline_refusal_scores_harmful.mean().item(),
        token_labels=token_labels,
        title='LOCAL ablating direction on harmful instructions (source layer only)',
        artifact_dir=artifact_dir,
        artifact_name='ablation_scores_local'
    )
    plot_refusal_scores(
        refusal_scores=ablation_kl_div_scores_local,
        baseline_refusal_score=0.0,
        token_labels=token_labels,
        title='LOCAL KL Divergence when ablating direction (source layer only)',
        artifact_dir=artifact_dir,
        artifact_name='kl_div_scores_local'
    )
    plot_refusal_scores(
        refusal_scores=steering_median_scores_local,
        baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
        token_labels=token_labels,
        title=f'LOCAL steering median (source layer actadd on harmless, multipliers={actadd_multipliers})',
        artifact_dir=artifact_dir,
        artifact_name='steering_median_scores_local'
    )

    # --- Build scored lists ---
    all_scored = []
    filtered_scored = []
    json_output_all_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):
            refusal_score_val = ablation_refusal_scores[source_pos, source_layer].item()
            kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()
            refusal_score_local = ablation_refusal_scores_local[source_pos, source_layer].item()
            kl_div_score_local = ablation_kl_div_scores_local[source_pos, source_layer].item()
            steering_median = steering_median_scores[source_pos, source_layer].item()
            steering_median_local = steering_median_scores_local[source_pos, source_layer].item()

            sorting_score = -refusal_score_val
            sorting_score_local = -refusal_score_local

            row = {
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': refusal_score_val,
                'steering_median_score': steering_median,
                'steering_median_score_local': steering_median_local,
                'kl_div_score': kl_div_score,
                'sorting_score': sorting_score,
                'refusal_score_local': refusal_score_local,
                'kl_div_score_local': kl_div_score_local,
                'sorting_score_local': sorting_score_local,
            }

            json_output_all_scores.append(row)
            all_scored.append(row)

            discard_direction = filter_fn(
                refusal_score=refusal_score_local,
                steering_score=steering_median_local,
                kl_div_score=kl_div_score_local,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage
            )
            if not discard_direction:
                filtered_scored.append(row)

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(_sanitize_for_json(json_output_all_scores), f, indent=4)

    filtered_local_sorted = sorted(filtered_scored, key=_nan_safe_sort_key)
    with open(f"{artifact_dir}/direction_evaluations_filtered_local.json", 'w') as f:
        json.dump(_sanitize_for_json(filtered_local_sorted), f, indent=4)

    if compare_rankings:
        filtered_global_sorted = sorted(
            filtered_scored,
            key=lambda x: (-x['sorting_score'], x['position'], x['layer'])
        )
        with open(f"{artifact_dir}/direction_evaluations_filtered.json", 'w') as f:
            json.dump(_sanitize_for_json(filtered_global_sorted), f, indent=4)

    all_scored.sort(key=_nan_safe_sort_key)
    filtered_scored.sort(key=_nan_safe_sort_key)

    assert len(filtered_scored) > 0, "All scores have been filtered out!"

    best = filtered_scored[0]
    pos = best['position']
    layer = best['layer']
    top_direction_norm = torch.norm(candidate_directions[pos, layer]).item()

    print(f"Selected direction: position={pos}, layer={layer}")
    print(f"Ablation refusal score (local): {best['refusal_score_local']:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"Steering median (local): {best['steering_median_score_local']:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
    print(f"KL Divergence (local): {best['kl_div_score_local']:.4f}")
    print(f"Top direction norm: {top_direction_norm:.4f}")
    print(f"Pool sizes: all={len(all_scored)}, filtered={len(filtered_scored)}")

    if compare_rankings:
        from scipy.stats import spearmanr

        global_ranking_order = sorted(
            filtered_scored,
            key=lambda x: (-x['sorting_score'], x['position'], x['layer'])
        )
        global_ranking = [(r['position'], r['layer']) for r in global_ranking_order]
        local_ranking = [(r['position'], r['layer']) for r in filtered_scored]

        global_rank_map = {comp: rank for rank, comp in enumerate(global_ranking)}
        local_rank_map = {comp: rank for rank, comp in enumerate(local_ranking)}

        components = list(global_rank_map.keys())
        global_ranks = [global_rank_map[c] for c in components]
        local_ranks = [local_rank_map[c] for c in components]

        if len(components) >= 2:
            rho, p_value = spearmanr(global_ranks, local_ranks)
            print(f"\n--- Ranking comparison: all-layer vs local (per-component) ablation ---")
            print(f"Spearman rho = {rho:.4f} (p = {p_value:.2e}), n = {len(components)} filtered components")
            print(f"Global top-5: {global_ranking[:5]}")
            print(f"Local  top-5: {local_ranking[:5]}")
            top1_match = global_ranking[0] == local_ranking[0]
            top5_overlap = len(set(global_ranking[:5]) & set(local_ranking[:5]))
            print(f"Top-1 match: {top1_match}, Top-5 overlap: {top5_overlap}/5")
        else:
            print(f"\nOnly {len(components)} filtered component(s) — cannot compute rank correlation.")

    return all_scored, filtered_scored, top_direction_norm


def select_direction(
    model_base: ModelBase,
    harmful_instructions,
    harmless_instructions,
    candidate_directions: Float[Tensor, 'n_pos n_layer d_model'],
    artifact_dir,
    actadd_multipliers: list,
    kl_threshold=0.1,
    induce_refusal_threshold=0.0,
    prune_layer_percentage=0.2,
    batch_size=32,
):
    """Backward-compatible wrapper that returns only the best component."""
    _all_ranked, filtered_ranked, top_direction_norm = select_direction_ranked(
        model_base=model_base,
        harmful_instructions=harmful_instructions,
        harmless_instructions=harmless_instructions,
        candidate_directions=candidate_directions,
        artifact_dir=artifact_dir,
        actadd_multipliers=actadd_multipliers,
        kl_threshold=kl_threshold,
        induce_refusal_threshold=induce_refusal_threshold,
        prune_layer_percentage=prune_layer_percentage,
        batch_size=batch_size,
    )

    best = filtered_ranked[0]
    pos = best['position']
    layer = best['layer']
    return pos, layer, candidate_directions[pos, layer]

def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if mask is None:
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean

def kl_div_fn(
    logits_a: Float[Tensor, 'batch seq_pos d_vocab'],
    logits_b: Float[Tensor, 'batch seq_pos d_vocab'],
    mask: Int[Tensor, "batch seq_pos"]=None,
    epsilon: Float=1e-6
) -> Float[Tensor, 'batch']:
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)

    if mask is None:
        return torch.mean(kl_divs, dim=-1)
    else:
        return masked_mean(kl_divs, mask).mean(dim=-1)
