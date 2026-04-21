"""INLP-based refusal direction extraction.

For each (position, layer) pair, trains an Iterative Nullspace Projection (INLP)
classifier to separate harmful from harmless activations at that layer.

Two outputs are produced:
  inlp_directions  — shape (n_pos, n_layers, d_model): the most discriminative
                     direction per (pos, layer), usable as a drop-in replacement
                     for mean-difference candidate directions in select_direction().
  nullspace P      — a (d_model, d_model) numpy matrix for the *selected*
                     (pos, layer) that projects activations into the subspace
                     orthogonal to ALL iteratively found refusal directions.

Linear classifiers are trained with PyTorch LBFGS on GPU (falls back to CPU).
No sklearn or scipy dependency.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks, get_nullspace_projection_input_pre_hook, get_activation_addition_input_pre_hook
from pipeline.model_utils.model_base import ModelBase


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _split_train_val(n_samples: int, val_frac: float = 0.3, seed: int = 42):
    """Split indices into train/val sets (deterministic)."""
    n_val = max(4, int(n_samples * val_frac))
    rng = np.random.default_rng(seed=seed)
    idx = rng.permutation(n_samples)
    return idx[n_val:], idx[:n_val]


# ─── INLP core ────────────────────────────────────────────────────────────────

def _get_rowspace_projection(W: torch.Tensor) -> torch.Tensor:
    """W: (1, d) → (d, d) rowspace projection matrix, on same device as W."""
    if W.norm() < 1e-9:
        return torch.zeros(W.shape[-1], W.shape[-1], device=W.device, dtype=W.dtype)
    _, _, Vh = torch.linalg.svd(W, full_matrices=False)  # Vh: (1, d)
    return Vh.T @ Vh  # (d, d)


def _run_inlp(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_dev: np.ndarray,
    Y_dev: np.ndarray,
    device: torch.device,
    n_classifiers: int = 100,
    min_accuracy: float = 0.55,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[float], np.ndarray]:
    """Run INLP using PyTorch logistic regression on GPU.

    Returns
    -------
    P : np.ndarray, shape (d, d)
        Nullspace projection matrix (projects out all refusal directions found).
    first_dir : np.ndarray, shape (1, d) or None
        First classifier direction as a unit vector (oriented: harmful > harmless),
        or None if no classifier exceeded min_accuracy.
    accuracies : list[float]
        Dev-set accuracy at each INLP iteration.
    classifier_dirs : np.ndarray, shape (n_classifiers, d)
        Unit-normalized classifier direction at each iteration, oriented so that
        harmful activations score higher than harmless. These are the individual
        directions projected out across iterations; any subset S gives a valid
        projection P_S = I - proj(span(classifier_dirs[S])).
    """
    d = X_train.shape[-1]
    dtype = torch.float32

    Xtr = torch.from_numpy(X_train).to(device=device, dtype=dtype)
    Ytr = torch.from_numpy(Y_train).to(device=device, dtype=dtype)
    Xdv = torch.from_numpy(X_dev).to(device=device, dtype=dtype)
    Ydv = torch.from_numpy(Y_dev)

    first_dir: Optional[torch.Tensor] = None
    accuracies: List[float] = []
    classifier_dirs: List[torch.Tensor] = []

    Xtr_proj = Xtr.clone()
    Xdv_proj = Xdv.clone()
    P_current = torch.eye(d, device=device, dtype=dtype)
    # Running sum of rowspace projections — avoids O(k²) memory from re-stacking.
    Q = torch.zeros(d, d, device=device, dtype=dtype)
    lam = 1.0 / (2 * 0.1 * len(Ytr))

    for _ in range(n_classifiers):
        W = torch.zeros(d, device=device, dtype=dtype, requires_grad=True)
        b = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=200, tolerance_grad=1e-5)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(Xtr_proj @ W + b, Ytr)
            loss = loss + lam * W.pow(2).sum()
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            preds = (Xdv_proj @ W + b > 0).cpu()
            acc = (preds == Ydv.bool()).float().mean().item()

        Wmat = W.detach().unsqueeze(0)  # (1, d)

        # Orient so harmful > harmless and unit-normalize.
        scores = Xtr @ Wmat.T
        if scores[Ytr == 1].mean() < scores[Ytr == 0].mean():
            Wmat = -Wmat
        norm = Wmat.norm()
        Wunit = Wmat / (norm + 1e-12)
        classifier_dirs.append(Wunit.squeeze(0))

        if first_dir is None:
            first_dir = Wunit

        accuracies.append(acc)
        Q = Q + _get_rowspace_projection(Wunit)

        svdvals = torch.linalg.svdvals(Q)
        _, _, Vh = torch.linalg.svd(Q, full_matrices=False)
        rank = int((svdvals > 1e-7).sum())
        P_current = torch.eye(d, device=device, dtype=dtype) - Vh[:rank].T @ Vh[:rank]

        if acc < min_accuracy:
            break

        if len(accuracies) >= 3 and accuracies[-1] == accuracies[-2] == accuracies[-3]:
            break

        Xtr_proj = (P_current @ Xtr.T).T
        Xdv_proj = (P_current @ Xdv.T).T

    if accuracies:
        print(f"INLP: {len(accuracies)} classifiers, accuracies = {[f'{a:.3f}' for a in accuracies]}")

    if classifier_dirs:
        classifier_dirs_arr = torch.stack(classifier_dirs, dim=0).cpu().numpy()
    else:
        classifier_dirs_arr = np.zeros((0, d), dtype=np.float32)
    first_dir_np = first_dir.cpu().numpy() if first_dir is not None else None
    return P_current.cpu().numpy(), first_dir_np, accuracies, classifier_dirs_arr


# ─── Activation extraction ────────────────────────────────────────────────────

def get_all_activations(
    model,
    instructions: List[str],
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    batch_size: int = 32,
    positions: List[int] = [-1],
) -> Float[Tensor, "n_instructions n_positions n_layers d_model"]:
    """Extract per-instruction residual-stream activations at every layer.

    Parameters
    ----------
    positions : list[int]
        Sequence-position indices (negative = from end) to extract from.

    Returns
    -------
    Tensor of shape (n_instructions, n_positions, n_layers, d_model), float32, on CPU.
    """
    torch.cuda.empty_cache()
    n_layers = len(block_modules)
    cache: List[List[Tensor]] = [[] for _ in range(n_layers)]

    def _make_pre_hook(layer_idx: int):
        def hook_fn(module, input):
            act = input[0].detach().cpu().float()   # (batch, seq_len, d_model)
            cache[layer_idx].append(act[:, positions, :])  # (batch, n_pos, d_model)
        return hook_fn

    fwd_pre_hooks = [(block_modules[l], _make_pre_hook(l)) for l in range(n_layers)]

    for i in tqdm(range(0, len(instructions), batch_size), desc="Extracting activations"):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i + batch_size])
        with torch.no_grad(), add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    # Stack: (n_inst, n_pos, d_model) per layer → (n_inst, n_pos, n_layers, d_model)
    per_layer = [torch.cat(cache[l], dim=0) for l in range(n_layers)]
    return torch.stack(per_layer, dim=2)


# ─── Main direction extraction ────────────────────────────────────────────────

def generate_directions_inlp(
    model_base: ModelBase,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    artifact_dir: str,
    batch_size: int = 32,
    n_classifiers: int = 100,
    min_accuracy: float = 0.55,
    val_frac: float = 0.3,
    prune_layer_percentage: Optional[float] = 0.20,
    offload_model: bool = False,
) -> None:
    """Extract activations and compute INLP for all (pos, layer) pairs.

    Saves to ``artifact_dir``:
      - ``harmful_activations.pt`` / ``harmless_activations.pt`` (raw activations)
      - ``inlp_results.pt`` — per-(pos, layer) nullspace projection P,
        first classifier direction, and accuracies.

    ``prune_layer_percentage`` skips INLP training for the last fraction of
    layers, matching the filter applied by downstream selection. Pass ``None``
    to train every layer. The activation cache (harmful/harmless .pt) is
    always written in full, so changing this value on a later run re-uses
    cached activations and only re-trains INLP.
    """
    os.makedirs(artifact_dir, exist_ok=True)

    positions = list(range(-len(model_base.eoi_toks), 0))

    harmful_cache = os.path.join(artifact_dir, "harmful_activations.pt")
    harmless_cache = os.path.join(artifact_dir, "harmless_activations.pt")

    from_cache = os.path.exists(harmful_cache) and os.path.exists(harmless_cache)

    if from_cache:
        print(f"INLP: loading cached activations from {artifact_dir}")
        harmful_acts = torch.load(harmful_cache, map_location="cpu")
        harmless_acts = torch.load(harmless_cache, map_location="cpu")
    else:
        print("INLP: extracting harmful activations …")
        harmful_acts = get_all_activations(
            model_base.model, harmful_instructions,
            model_base.tokenize_instructions_fn, model_base.model_block_modules,
            batch_size=batch_size, positions=positions,
        )   # (n_harmful, n_pos, n_layers, d_model)

        print("INLP: extracting harmless activations …")
        harmless_acts = get_all_activations(
            model_base.model, harmless_instructions,
            model_base.tokenize_instructions_fn, model_base.model_block_modules,
            batch_size=batch_size, positions=positions,
        )   # (n_harmless, n_pos, n_layers, d_model)

        torch.save(harmful_acts, harmful_cache)
        torch.save(harmless_acts, harmless_cache)

    # Run INLP for every (pos, layer) pair
    n_pos = len(positions)
    n_layers = model_base.model.config.num_hidden_layers
    n_min = min(harmful_acts.shape[0], harmless_acts.shape[0])
    harmful_acts = harmful_acts[:n_min]
    harmless_acts = harmless_acts[:n_min]

    Y = np.array([1] * n_min + [0] * n_min, dtype=np.int32)
    train_idx, val_idx = _split_train_val(2 * n_min, val_frac=val_frac)

    model_original_device = next(model_base.model.parameters()).device
    should_offload = offload_model and from_cache and model_original_device.type == "cuda"
    if should_offload:
        print(f"INLP: offloading model to CPU to free GPU memory for INLP tensors")
        model_base.model.to("cpu")
        torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inlp_params = {}

    if prune_layer_percentage is None:
        layers_to_train = list(range(n_layers))
    else:
        cutoff = int(n_layers * (1.0 - prune_layer_percentage))
        layers_to_train = list(range(cutoff))
        n_pruned = n_layers - cutoff
        if n_pruned > 0:
            print(
                f"INLP: pruning last {n_pruned}/{n_layers} layers "
                f"(prune_layer_percentage={prune_layer_percentage}); "
                f"training layers 0..{cutoff - 1}"
            )

    try:
        for pos_idx in range(n_pos):
            src_pos = pos_idx - n_pos
            for layer_idx in tqdm(layers_to_train,
                                   desc=f"INLP generate (pos {src_pos})"):
                X = torch.cat([
                    harmful_acts[:, pos_idx, layer_idx, :],
                    harmless_acts[:, pos_idx, layer_idx, :],
                ], dim=0).numpy()

                P, first_dir, accuracies, classifier_dirs = _run_inlp(
                    X[train_idx], Y[train_idx],
                    X[val_idx], Y[val_idx],
                    device=device,
                    n_classifiers=n_classifiers,
                    min_accuracy=min_accuracy,
                )

                inlp_params[f"({pos_idx}, {layer_idx})"] = {
                    "P": P,
                    "first_dir": first_dir.squeeze() if first_dir is not None else None,
                    "accuracies": accuracies,
                    "classifier_dirs": classifier_dirs,
                }
    finally:
        if should_offload:
            print(f"INLP: restoring model to {model_original_device}")
            model_base.model.to(model_original_device)

    results = {
        "inlp_params": inlp_params,
        "positions": positions,
        "n_layers": n_layers,
        "n_classifiers": n_classifiers,
        "min_accuracy": min_accuracy,
        "val_frac": val_frac,
        "prune_layer_percentage": prune_layer_percentage,
    }
    torch.save(results, os.path.join(artifact_dir, "inlp_results.pt"))
    print(f"INLP: saved results for {len(inlp_params)} (pos, layer) pairs to {artifact_dir}/inlp_results.pt")


# ─── Extract directions from a P matrix ──────────────────────────────────────

def get_directions_from_P(
    P: np.ndarray,
    k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract INLP directions from projection matrix P via SVD of (I - P).

    Since P = I - V^T @ V where V contains orthogonal INLP directions,
    we can recover V from SVD(I - P) and optionally keep only the top-k
    singular vectors to build a k-restricted projection.

    Parameters
    ----------
    P : np.ndarray, shape (d, d)
        Nullspace projection matrix.
    k : int or None
        If given, keep only the top-k singular vectors of (I - P).
        None keeps all directions found in P (full rank of I - P).

    Returns
    -------
    directions : np.ndarray, shape (rank_or_k, d)
        Orthogonal directions recovered from P.
    P_k : np.ndarray, shape (d, d)
        Rebuilt projection using only top-k directions:
        P_k = I - V[:k].T @ V[:k].
        When k is None, P_k reproduces the original P up to numerical error.
    """
    d = P.shape[0]
    # Use float64 so k-restricted reconstructions are as stable as possible.
    I_minus_P = np.eye(d, dtype=np.float64) - P.astype(np.float64)
    _, s, Vh = np.linalg.svd(I_minus_P, full_matrices=False)
    rank = int((s > 1e-6).sum())
    if k is not None:
        rank = min(k, rank)
    directions = Vh[:rank]                               # (rank, d)
    P_k = np.eye(d, dtype=np.float64) - directions.T @ directions  # (d, d)
    return directions, P_k


# ─── Nullspace projection for selected (pos, layer) ──────────────────────────

def compute_inlp_nullspace_projection(
    artifact_dir: str,
    model_base: ModelBase,
    pos: int,
    layer: int,
) -> np.ndarray:
    """Load the pre-computed nullspace projection P for the given (pos, layer).

    Parameters
    ----------
    pos : int
        Negative position index (e.g. -1 for last EOI token).
    layer : int
        Layer index (0-indexed).

    Returns
    -------
    P : np.ndarray, shape (d_model, d_model), dtype float32
        Nullspace projection matrix that removes all linearly separable
        refusal-related information found at this (pos, layer).
    """
    results_path = os.path.join(artifact_dir, "inlp_results.pt")
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"INLP results not found at {results_path}. "
            "Run generate_directions_inlp first."
        )

    results = torch.load(results_path, map_location="cpu")
    positions = results["positions"]
    n_pos = len(positions)
    pos_idx = pos + n_pos  # e.g. pos=-1, n_pos=2 → pos_idx=1

    key = f"({pos_idx}, {layer})"
    if key not in results["inlp_params"]:
        raise KeyError(f"No INLP result for pos_idx={pos_idx}, layer={layer}")

    entry = results["inlp_params"][key]
    accuracies = entry["accuracies"]
    print(
        f"Nullspace projection: {len(accuracies)} classifiers removed, "
        f"accuracies: {[f'{a:.3f}' for a in accuracies]}"
    )

    return entry["P"]


# ─── Direction selection via nullspace projection effect ──────────────────────

def resolve_indices_for_policy(
    policy: str,
    k_fixed: Optional[int],
    accuracies: List[float],
) -> Optional[List[int]]:
    """Map ``(policy, k_fixed, accuracies)`` to the classifier indices to keep.

    Returns ``None`` when ``policy == 'none'`` (caller keeps the full P).
    Returns a list of classifier indices otherwise. Note that accuracies are
    NOT monotone in INLP — a policy like ``acc80`` yields the (possibly
    non-contiguous) set of indices whose dev accuracy meets the threshold.
    Callers should treat an empty list as "no valid projection for this
    (pos, layer)" and skip it.
    """
    p = (policy or 'none').lower()
    if p == 'none':
        return None
    if p == 'fixed':
        if k_fixed is None:
            raise ValueError("inlp_k_policy='fixed' requires inlp_k_restrict to be set")
        k = int(k_fixed)
        return list(range(min(k, len(accuracies))))
    if p == 'acc90':
        return [i for i, a in enumerate(accuracies) if a >= 0.90]
    if p == 'acc80':
        return [i for i, a in enumerate(accuracies) if a >= 0.80]
    raise ValueError(f"Unknown inlp_k_policy: {policy!r}")


def build_P_from_indices(
    classifier_dirs: np.ndarray,
    indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a nullspace projection P removing the subspace spanned by the
    selected classifier directions.

    Parameters
    ----------
    classifier_dirs : np.ndarray, shape (n_classifiers, d)
        Unit-normalized INLP classifier directions (one per iteration).
    indices : list[int]
        Subset of classifier indices to project out.

    Returns
    -------
    selected : np.ndarray, shape (len(indices), d)
        The selected classifier directions (copy).
    P : np.ndarray, shape (d, d), dtype float64
        Projection matrix P = I - V_orth^T @ V_orth where V_orth is an
        orthonormal basis of span(selected).  When ``indices`` is empty,
        P is the identity.
    """
    d = classifier_dirs.shape[-1]
    if len(indices) == 0:
        return np.zeros((0, d), dtype=np.float64), np.eye(d, dtype=np.float64)
    selected = classifier_dirs[np.asarray(indices, dtype=np.int64)].astype(np.float64)
    # Orthonormal basis of span(selected); V_orth rows are orthonormal.
    _, s, Vh = np.linalg.svd(selected, full_matrices=False)
    rank = int((s > 1e-7).sum())
    V_orth = Vh[:rank]
    P = np.eye(d, dtype=np.float64) - V_orth.T @ V_orth
    return selected, P


def _k_suffix_for_policy(policy: str, k_fixed: Optional[int]) -> str:
    p = (policy or 'none').lower()
    if p == 'none':
        return ''
    if p == 'fixed':
        return f"_k{k_fixed}" if k_fixed is not None else "_kna"
    return f"_{p}"


def select_direction_inlp_ranked(
    artifact_dir: str,
    model_base: ModelBase,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    actadd_multipliers: list,
    direction_norm: float,
    kl_threshold: Optional[float] = 0.1,
    prune_layer_percentage: float = 0.20,
    batch_size: int = 32,
    k_policy: str = 'none',
    k_fixed: Optional[int] = None,
) -> List[dict]:
    """Select the best (pos, layer) for INLP using actadd-median refusal scoring.

    Loads pre-computed INLP results from ``inlp_results.pt`` (saved by
    ``generate_directions_inlp``), then scores each (pos, layer) by sweeping
    actadd multipliers scaled by ``direction_norm`` (from the top mean-diff
    direction) and taking the median refusal score.

    Parameters
    ----------
    artifact_dir : str
        Directory with ``inlp_results.pt`` saved by ``generate_directions_inlp``.
    actadd_multipliers : list
        Multipliers for the actadd coefficient sweep.
    direction_norm : float
        Norm of the top mean-diff direction, used to scale INLP actadd
        coefficients for fair comparison.
    kl_threshold : float or None
        Reject (pos, layer) if the KL divergence between baseline and P-projected
        harmless logits exceeds this value.  None disables KL filtering.
    prune_layer_percentage : float
        Skip the last fraction of layers (same convention as select_direction).

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(all_scores, filtered_scores)`` — both ranked by sorting_score.
        All entries carry first_dir and P.  ``all_scores`` is the full
        unfiltered pool (for top-k layer selection); ``filtered_scores``
        is the filtered pool (for best direction selection).
    """
    import json as _json
    from pipeline.submodules.select_direction import (
        get_refusal_scores, kl_div_fn, get_last_position_logits, filter_fn, plot_refusal_scores,
    )

    results_path = os.path.join(artifact_dir, "inlp_results.pt")
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"INLP results not found at {results_path}. "
            "Run generate_directions_inlp first."
        )

    results = torch.load(results_path, map_location="cpu")
    inlp_params = results["inlp_params"]
    positions = results["positions"]
    n_pos = len(positions)
    n_layers = results["n_layers"]

    device = model_base.model.device

    # Baselines for reporting and filtering parity with mean-diff selection.
    baseline_refusal_harmful = get_refusal_scores(
        model_base.model, harmful_instructions,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
        batch_size=batch_size,
    ).mean().item()

    baseline_refusal_harmless = get_refusal_scores(
        model_base.model, harmless_instructions,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
        batch_size=batch_size,
    ).mean().item()

    # Baseline logits on harmless instructions for KL filtering
    baseline_harmless_logits = None
    if kl_threshold is not None:
        baseline_harmless_logits = get_last_position_logits(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            instructions=harmless_instructions,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            batch_size=batch_size,
        )

    all_scores: list = []
    filtered_scores: list = []

    # Score tensors for plotting (NaN = pruned/missing)
    plot_refusal_scores_tensor = torch.full((n_pos, n_layers), float('nan'))
    plot_steering_scores_tensor = torch.full((n_pos, n_layers), float('nan'))
    plot_kl_div_scores_tensor = torch.full((n_pos, n_layers), float('nan'))

    for pos_idx in range(n_pos):
        src_pos = pos_idx - n_pos  # convert to negative index
        for layer_idx in tqdm(range(n_layers),
                               desc=f"INLP selection (pos {src_pos})"):

            key = f"({pos_idx}, {layer_idx})"
            if key not in inlp_params:
                continue

            entry = inlp_params[key]
            P_full = entry["P"]
            first_dir = entry["first_dir"]
            accuracies = entry["accuracies"]
            classifier_dirs = entry.get("classifier_dirs")

            if first_dir is None or len(accuracies) == 0:
                continue

            # Resolve which classifier indices to project out for this policy.
            # Note: accuracies are not monotone, so acc-thresholded policies
            # may select a non-contiguous subset (e.g. {0, 1, 3, 4}).
            indices = resolve_indices_for_policy(k_policy, k_fixed, accuracies)
            if indices is None:
                P = P_full
                selected_accs = list(accuracies)
            else:
                if len(indices) == 0:
                    print(f"Skipping (pos {src_pos}, layer {layer_idx}): no classifiers pass the policy threshold (accuracies={accuracies})")
                    continue
                if classifier_dirs is None or classifier_dirs.shape[0] == 0:
                    print(f"Skipping (pos {src_pos}, layer {layer_idx}): classifier_dirs unavailable; re-run generate_directions_inlp")
                    continue
                _, P = build_P_from_indices(classifier_dirs, indices)
                selected_accs = [float(accuracies[i]) for i in indices]
            print(f"accuracies={accuracies}, selected_indices={indices}")

            # Refusal score: nullspace projection P on harmful
            nullspace_hooks = [(model_base.model_block_modules[layer_idx],
                                get_nullspace_projection_input_pre_hook(P))]
            refusal_score = get_refusal_scores(
                model_base.model, harmful_instructions,
                model_base.tokenize_instructions_fn, model_base.refusal_toks,
                fwd_pre_hooks=nullspace_hooks, fwd_hooks=[],
                batch_size=batch_size,
            ).mean().item()

            # Normalize INLP direction to unit and use direction_norm for fair scaling
            first_dir_tensor = torch.from_numpy(first_dir.squeeze()).float()
            first_dir_norm = torch.norm(first_dir_tensor).item()
            first_dir_unit = first_dir_tensor / (first_dir_norm + 1e-8)

            # Steering score: median actadd on harmless, sweep multipliers * direction_norm
            per_mult_scores = []
            for m in actadd_multipliers:
                coeff = m * direction_norm  # positive: induce refusal on harmless
                fwd_pre_hooks = [(model_base.model_block_modules[layer_idx],
                                  get_activation_addition_input_pre_hook(
                                      vector=first_dir_unit, coeff=coeff))]

                scores = get_refusal_scores(
                    model_base.model, harmless_instructions,
                    model_base.tokenize_instructions_fn, model_base.refusal_toks,
                    fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=[],
                    batch_size=batch_size,
                )
                per_mult_scores.append(scores.mean().item())

            steering_score = float(torch.tensor(per_mult_scores).max())

            # KL filtering: apply nullspace projection P, measure distortion on harmless
            nullspace_fwd_pre_hooks = [(model_base.model_block_modules[layer_idx],
                                        get_nullspace_projection_input_pre_hook(P))]
            kl = 0.0
            if baseline_harmless_logits is not None:
                intervention_logits = get_last_position_logits(
                    model=model_base.model,
                    tokenizer=model_base.tokenizer,
                    instructions=harmless_instructions,
                    tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                    fwd_pre_hooks=nullspace_fwd_pre_hooks,
                    fwd_hooks=[],
                    batch_size=batch_size,
                )
                kl = kl_div_fn(
                    baseline_harmless_logits, intervention_logits, mask=None
                ).mean().item()

            if indices is None:
                k_used = len(accuracies)
                indices_used = list(range(len(accuracies)))
            else:
                k_used = len(indices)
                indices_used = list(indices)
            row = {
                'position': src_pos,
                'layer': layer_idx,
                'refusal_score': refusal_score,
                'steering_score': steering_score,
                'kl_div_score': kl,
                'n_classifiers': len(accuracies),
                'first_classifier_acc': accuracies[0],
                'sorting_score': -refusal_score,
                'first_dir': first_dir.squeeze().copy(),
                'P': P.copy(),
                'k_used': k_used,
                'indices_used': indices_used,
                'k_accs_used': selected_accs,
            }

            all_scores.append(row)
            plot_refusal_scores_tensor[pos_idx, layer_idx] = refusal_score
            plot_steering_scores_tensor[pos_idx, layer_idx] = steering_score
            plot_kl_div_scores_tensor[pos_idx, layer_idx] = kl

            discard_direction = filter_fn(
                refusal_score=refusal_score,
                steering_score=steering_score,
                kl_div_score=kl,
                layer=layer_idx,
                n_layer=n_layers,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=0.0,
                prune_layer_percentage=prune_layer_percentage,
            )

            if not discard_direction:
                filtered_scores.append(row)

    k_suffix = _k_suffix_for_policy(k_policy, k_fixed)
    token_labels = model_base.tokenizer.batch_decode(model_base.eoi_toks)
    plot_refusal_scores(
        refusal_scores=plot_refusal_scores_tensor,
        baseline_refusal_score=baseline_refusal_harmful,
        token_labels=token_labels,
        title='INLP: nullspace projection refusal on harmful instructions',
        artifact_dir=artifact_dir,
        artifact_name=f'inlp_refusal_scores{k_suffix}',
    )
    plot_refusal_scores(
        refusal_scores=plot_steering_scores_tensor,
        baseline_refusal_score=baseline_refusal_harmless,
        token_labels=token_labels,
        title=f'INLP: steering median on harmless instructions (multipliers={actadd_multipliers})',
        artifact_dir=artifact_dir,
        artifact_name=f'inlp_steering_median_scores{k_suffix}',
    )
    plot_refusal_scores(
        refusal_scores=plot_kl_div_scores_tensor,
        baseline_refusal_score=0.0,
        token_labels=token_labels,
        title='INLP: KL divergence (nullspace projection on harmless)',
        artifact_dir=artifact_dir,
        artifact_name=f'inlp_kl_div_scores{k_suffix}',
    )
    all_scores.sort(key=lambda x: (-x['first_classifier_acc'], -x['sorting_score'], x['position'], x['layer']))
    if len(filtered_scores) == 0:
        if len(all_scores) == 0:
            print("WARNING: No valid INLP direction found at any (pos, layer). INLP interventions will be skipped.")
            return all_scores, []

        # Fallback: select the best available INLP direction from the unfiltered,
        # non-pruned pool using composite ranking.
        print("WARNING: No INLP direction passed filtering. Falling back to best unfiltered, non-pruned direction.")
        print(f"  • kl_threshold={kl_threshold}: {sum(1 for x in all_scores if x['kl_div_score'] > kl_threshold)} directions exceed KL limit")
        print(f"  • prune_layer_percentage={prune_layer_percentage}: {sum(1 for x in all_scores if x['layer'] >= int(n_layers * (1.0 - prune_layer_percentage)))} directions in pruned layers")

        fallback_pool = [
            x for x in all_scores
            if prune_layer_percentage is None or x['layer'] < int(n_layers * (1.0 - prune_layer_percentage))
        ]
        if len(fallback_pool) == 0:
            print("WARNING: No unfiltered INLP direction remains after applying the layer-pruning fallback constraint. INLP interventions will be skipped.")
            return all_scores, []

        # Rank by: lowest refusal_score, then lowest kl_div_score, then best accuracy,then highest steering_score
        fallback = sorted(
            fallback_pool,
            key=lambda x: (-x['first_classifier_acc'], x['refusal_score'], x['kl_div_score'], -x['steering_score']),
        )
        filtered_scores.append(fallback[0])
        print(
            f"  Fallback selected: pos={fallback[0]['position']}, layer={fallback[0]['layer']}, "
            f"refusal={fallback[0]['refusal_score']:.4f}, kl={fallback[0]['kl_div_score']:.4f}, "
            f"steering={fallback[0]['steering_score']:.4f}, accuracy={fallback[0]['first_classifier_acc']:.4f}"
        )

    filtered_scores.sort(key=lambda x: (-x['first_classifier_acc'],-x['sorting_score'], x['position'], x['layer']))

    def _json_row(x):
        return {
            'position': x['position'],
            'layer': x['layer'],
            'refusal_score': x['refusal_score'],
            'steering_score': x['steering_score'],
            'kl_div_score': x['kl_div_score'],
            'n_classifiers': x['n_classifiers'],
            'first_classifier_acc': x['first_classifier_acc'],
            'sorting_score': x['sorting_score'],
            'k_used': x.get('k_used'),
            'k_accs_used': x.get('k_accs_used'),
        }

    with open(os.path.join(artifact_dir, f"inlp_selection_scores{k_suffix}.json"), "w") as f:
        _json.dump([_json_row(x) for x in all_scores], f, indent=4)
    with open(os.path.join(artifact_dir, f"inlp_selection_scores_filtered{k_suffix}.json"), "w") as f:
        _json.dump([_json_row(x) for x in filtered_scores], f, indent=4)

    best = filtered_scores[0]

    print(
        f"INLP selection: best pos={best['position']}, layer={best['layer']}, "
        f"refusal_score={best['refusal_score']:.4f} "
        f"(harmful baseline={baseline_refusal_harmful:.4f}, "
        f"harmless steering baseline={baseline_refusal_harmless:.4f}, "
        f"harmless steering={best['steering_score']:.4f}, "
        f"kl={best['kl_div_score']:.4f}), "
        f"first_classifier_acc={best['first_classifier_acc']:.4f}, "
        f"k_used={best.get('k_used')}, "
        f"k_policy={k_policy}, k_fixed={k_fixed}"
    )
    print(f"INLP pool sizes: all={len(all_scores)}, filtered={len(filtered_scores)}")

    return all_scores, filtered_scores


def select_direction_inlp(
    artifact_dir: str,
    model_base: ModelBase,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    actadd_multipliers: list,
    direction_norm: float,
    kl_threshold: Optional[float] = 0.1,
    prune_layer_percentage: float = 0.20,
    batch_size: int = 32,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper that returns only the best component."""
    ranked = select_direction_inlp_ranked(
        artifact_dir=artifact_dir,
        model_base=model_base,
        harmful_instructions=harmful_instructions,
        harmless_instructions=harmless_instructions,
        actadd_multipliers=actadd_multipliers,
        direction_norm=direction_norm,
        kl_threshold=kl_threshold,
        prune_layer_percentage=prune_layer_percentage,
        batch_size=batch_size,
    )

    best = ranked[0]
    return best['position'], best['layer'], best['first_dir'], best['P']
