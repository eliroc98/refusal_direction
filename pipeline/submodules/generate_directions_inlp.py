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

from pipeline.utils.hook_utils import add_hooks, get_nullspace_projection_input_pre_hook
from pipeline.model_utils.model_base import ModelBase


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
    n_classifiers: int = 20,
    min_accuracy: float = 0.55,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
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
    """
    d = X_train.shape[-1]
    dtype = torch.float32

    Xtr = torch.from_numpy(X_train).to(device=device, dtype=dtype)
    Ytr = torch.from_numpy(Y_train).to(device=device, dtype=dtype)
    Xdv = torch.from_numpy(X_dev).to(device=device, dtype=dtype)
    Ydv = torch.from_numpy(Y_dev)  # CPU for accuracy check

    rowspace_projs: List[torch.Tensor] = []
    first_dir: Optional[np.ndarray] = None
    accuracies: List[float] = []

    Xtr_proj = Xtr.clone()
    Xdv_proj = Xdv.clone()
    P_current = torch.eye(d, device=device, dtype=dtype)
    lam = 1.0 / (2 * 0.1 * len(Ytr))  # L2 reg equivalent to sklearn C=0.1

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

        if first_dir is None:
            scores = Xtr @ Wmat.T
            if scores[Ytr == 1].mean() < scores[Ytr == 0].mean():
                Wmat = -Wmat
            norm = Wmat.norm()
            first_dir = (Wmat / norm).cpu().numpy()

        accuracies.append(acc)
        rowspace_projs.append(_get_rowspace_projection(Wmat))

        Q = torch.stack(rowspace_projs).sum(dim=0)
        svdvals = torch.linalg.svdvals(Q)
        _, _, Vh = torch.linalg.svd(Q, full_matrices=False)
        rank = int((svdvals > 1e-7).sum())
        P_current = torch.eye(d, device=device, dtype=dtype) - Vh[:rank].T @ Vh[:rank]

        if acc < min_accuracy:
            break

        Xtr_proj = (P_current @ Xtr.T).T
        Xdv_proj = (P_current @ Xdv.T).T

    if accuracies:
        print(f"INLP: {len(accuracies)} classifiers, accuracies = {[f'{a:.3f}' for a in accuracies]}")

    return P_current.cpu().numpy(), first_dir, accuracies


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
) -> None:
    """Extract and cache activations for later INLP direction finding.

    Saves ``harmful_activations.pt`` and ``harmless_activations.pt`` to
    ``artifact_dir`` so that ``compute_inlp_nullspace_projection`` and
    ``select_direction_inlp`` can reuse them without re-running the model.
    """
    os.makedirs(artifact_dir, exist_ok=True)

    positions = list(range(-len(model_base.eoi_toks), 0))

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

    torch.save(harmful_acts,  os.path.join(artifact_dir, "harmful_activations.pt"))
    torch.save(harmless_acts, os.path.join(artifact_dir, "harmless_activations.pt"))


# ─── Nullspace projection for selected (pos, layer) ──────────────────────────

def compute_inlp_nullspace_projection(
    artifact_dir: str,
    model_base: ModelBase,
    pos: int,
    layer: int,
    n_classifiers: int = 20,
    min_accuracy: float = 0.55,
    val_frac: float = 0.2,
) -> np.ndarray:
    """Run full INLP for the selected (pos, layer) and return the nullspace P.

    Loads activations saved by ``generate_directions_inlp`` to avoid re-running
    the model.

    Parameters
    ----------
    pos : int
        Negative position index (e.g. -1 for last EOI token).
    layer : int
        Layer index (0-indexed).
    n_classifiers : int
        Maximum number of INLP iterations (directions to remove).

    Returns
    -------
    P : np.ndarray, shape (d_model, d_model), dtype float32
        Nullspace projection matrix that removes all linearly separable
        refusal-related information found at this (pos, layer).
    """
    harmful_path  = os.path.join(artifact_dir, "harmful_activations.pt")
    harmless_path = os.path.join(artifact_dir, "harmless_activations.pt")

    if not (os.path.exists(harmful_path) and os.path.exists(harmless_path)):
        raise FileNotFoundError(
            f"Activation files not found in {artifact_dir}. "
            "Run generate_directions_inlp first."
        )

    harmful_acts  = torch.load(harmful_path,  map_location="cpu")
    harmless_acts = torch.load(harmless_path, map_location="cpu")

    # Convert negative pos to list index
    positions = list(range(-len(model_base.eoi_toks), 0))
    n_pos     = len(positions)
    pos_idx   = pos + n_pos   # e.g. pos=-1, n_pos=2 → pos_idx=1

    n_min = min(harmful_acts.shape[0], harmless_acts.shape[0])
    harmful_acts  = harmful_acts[:n_min,  pos_idx, layer, :]  # (n_min, d_model)
    harmless_acts = harmless_acts[:n_min, pos_idx, layer, :]  # (n_min, d_model)

    Y = np.array([1] * n_min + [0] * n_min, dtype=np.int32)
    X = torch.cat([harmful_acts, harmless_acts], dim=0).numpy()  # float32

    n_val = max(4, int(2 * n_min * val_frac))
    rng   = np.random.default_rng(seed=42)
    idx   = rng.permutation(2 * n_min)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P, _, accuracies = _run_inlp(
        X[train_idx], Y[train_idx],
        X[val_idx],   Y[val_idx],
        device=device,
        n_classifiers=n_classifiers,
        min_accuracy=min_accuracy,
    )

    print(
        f"Nullspace projection: {len(accuracies)} classifiers removed, "
        f"accuracies: {[f'{a:.3f}' for a in accuracies]}"
    )

    return P


# ─── Direction selection via nullspace projection effect ──────────────────────

def select_direction_inlp(
    artifact_dir: str,
    model_base: ModelBase,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    n_classifiers: int = 20,
    min_accuracy: float = 0.55,
    val_frac: float = 0.2,
    kl_threshold: Optional[float] = 0.1,
    prune_layer_percentage: float = 0.20,
    batch_size: int = 32,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """Select the best (pos, layer) for INLP using nullspace projection effect.

    For each (pos, layer) pair, runs full INLP to obtain a nullspace projection
    matrix P, then measures how much P reduces refusal on harmful instructions.
    The (pos, layer) with the highest steering score (= baseline_refusal -
    projected_refusal) is selected.  The first INLP classifier direction from
    that winning run is returned as the representative direction for downstream
    activation-addition analysis.

    Loads activations cached by generate_directions_inlp.

    Parameters
    ----------
    artifact_dir : str
        Directory with 'harmful_activations.pt' and 'harmless_activations.pt'
        saved by generate_directions_inlp.
    n_classifiers : int
        Maximum INLP iterations per (pos, layer).
    kl_threshold : float or None
        Reject (pos, layer) if the KL divergence between baseline and P-projected
        harmless logits exceeds this value.  None disables KL filtering.
    prune_layer_percentage : float
        Skip the last fraction of layers (same convention as select_direction).

    Returns
    -------
    pos : int  (negative index)
    layer : int
    first_dir : np.ndarray, shape (d_model,)  — unit direction from best run
    P : np.ndarray, shape (d_model, d_model)  — nullspace projection from best run
    """
    import json as _json
    from pipeline.submodules.select_direction import (
        get_refusal_scores, kl_div_fn, get_last_position_logits,
    )

    harmful_path  = os.path.join(artifact_dir, "harmful_activations.pt")
    harmless_path = os.path.join(artifact_dir, "harmless_activations.pt")

    if not (os.path.exists(harmful_path) and os.path.exists(harmless_path)):
        raise FileNotFoundError(
            f"Activation files not found in {artifact_dir}. "
            "Run generate_directions_inlp first."
        )

    harmful_acts  = torch.load(harmful_path,  map_location="cpu")
    harmless_acts = torch.load(harmless_path, map_location="cpu")

    positions = list(range(-len(model_base.eoi_toks), 0))
    n_pos     = len(positions)
    n_layers  = model_base.model.config.num_hidden_layers

    n_min = min(harmful_acts.shape[0], harmless_acts.shape[0])
    harmful_acts  = harmful_acts[:n_min]
    harmless_acts = harmless_acts[:n_min]

    Y = np.array([1] * n_min + [0] * n_min, dtype=np.int32)
    n_val = max(4, int(2 * n_min * val_frac))
    rng   = np.random.default_rng(seed=42)
    idx   = rng.permutation(2 * n_min)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    device = model_base.model.device

    # Baseline refusal score on harmful instructions (no intervention)
    baseline_refusal = get_refusal_scores(
        model_base.model, harmful_instructions,
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

    best_steering_score = float('-inf')
    best_pos       = None
    best_layer     = None
    best_first_dir = None
    best_P         = None

    all_scores: list = []

    for pos_idx in range(n_pos):
        src_pos = pos_idx - n_pos  # convert to negative index
        for layer_idx in tqdm(range(n_layers),
                               desc=f"INLP selection (pos {src_pos})"):

            if (prune_layer_percentage is not None
                    and layer_idx >= int(n_layers * (1.0 - prune_layer_percentage))):
                continue

            X = torch.cat([
                harmful_acts[:,  pos_idx, layer_idx, :],
                harmless_acts[:, pos_idx, layer_idx, :],
            ], dim=0).numpy()  # (2*n_min, d_model), float32

            P, first_dir, accuracies = _run_inlp(
                X[train_idx], Y[train_idx],
                X[val_idx],   Y[val_idx],
                device=device,
                n_classifiers=n_classifiers,
                min_accuracy=min_accuracy,
            )

            if first_dir is None or len(accuracies) == 0:
                continue  # no classifier exceeded min_accuracy

            # Score: apply P only at the source layer (mirrors how select_direction
            # scores activation addition — single layer, not global).  Avoids the
            # compounding KL distortion that results from applying P at every layer.
            fwd_pre_hooks = [(model_base.model_block_modules[layer_idx],
                              get_nullspace_projection_input_pre_hook(P))]
            fwd_hooks = []
            projected_refusal = get_refusal_scores(
                model_base.model, harmful_instructions,
                model_base.tokenize_instructions_fn, model_base.refusal_toks,
                fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks,
                batch_size=batch_size,
            ).mean().item()

            # Higher steering_score = P suppresses more refusal = better direction
            steering_score = baseline_refusal - projected_refusal

            # KL filtering: reject if P distorts harmless generations too much
            if kl_threshold is not None and baseline_harmless_logits is not None:
                intervention_logits = get_last_position_logits(
                    model=model_base.model,
                    tokenizer=model_base.tokenizer,
                    instructions=harmless_instructions,
                    tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                    batch_size=batch_size,
                )
                import math as _math
                kl = kl_div_fn(
                    baseline_harmless_logits, intervention_logits, mask=None
                ).mean().item()
                if _math.isnan(kl) or kl > kl_threshold:
                    continue

            all_scores.append({
                'pos': src_pos,
                'layer': layer_idx,
                'steering_score': steering_score,
                'projected_refusal': projected_refusal,
                'n_classifiers': len(accuracies),
            })

            if steering_score > best_steering_score:
                best_steering_score = steering_score
                best_pos       = src_pos
                best_layer     = layer_idx
                best_first_dir = first_dir.squeeze()   # (d_model,)
                best_P         = P

    if best_pos is None:
        raise RuntimeError(
            "INLP selection: no valid direction found at any (pos, layer). "
            "Consider relaxing kl_threshold or prune_layer_percentage."
        )

    all_scores.sort(key=lambda x: x['steering_score'], reverse=True)
    with open(os.path.join(artifact_dir, "inlp_selection_scores.json"), "w") as f:
        _json.dump(all_scores, f, indent=4)

    print(
        f"INLP selection: best pos={best_pos}, layer={best_layer}, "
        f"steering_score={best_steering_score:.4f} "
        f"(baseline_refusal={baseline_refusal:.4f}, "
        f"projected_refusal={baseline_refusal - best_steering_score:.4f})"
    )

    return best_pos, best_layer, best_first_dir, best_P
