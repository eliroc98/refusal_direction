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

Dependencies: scikit-learn (LogisticRegression), scipy.linalg.orth.
No dependency on the number-words-constraint package.
"""

import os
import numpy as np
import scipy.linalg
import torch
from typing import List, Optional, Tuple

from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase


# ─── INLP core ────────────────────────────────────────────────────────────────

def _get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """Orthogonal projection matrix onto the row space of W."""
    if np.allclose(W, 0):
        return np.zeros((W.shape[-1], W.shape[-1]))
    basis = scipy.linalg.orth(W.T)
    basis *= np.sign(basis[0, 0])   # resolve sign ambiguity
    return basis @ basis.T


def _get_nullspace_projection(rowspace_projections: List[np.ndarray], d: int) -> np.ndarray:
    """Nullspace projection onto the intersection of all classifier nullspaces.

    Uses Ben-Israel (2013):  N(w1) ∩ … ∩ N(wn) = N(P_R(w1) + … + P_R(wn))
    """
    Q = np.sum(rowspace_projections, axis=0)
    return np.eye(d) - _get_rowspace_projection(Q)


def _run_inlp_sklearn(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_dev: np.ndarray,
    Y_dev: np.ndarray,
    n_classifiers: int = 20,
    min_accuracy: float = 0.55,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
    """Run INLP using logistic regression classifiers.

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
    from sklearn.linear_model import LogisticRegression

    d = X_train.shape[-1]
    rowspace_projections: List[np.ndarray] = []
    first_dir: Optional[np.ndarray] = None
    accuracies: List[float] = []

    X_tr = X_train.copy()
    X_dv = X_dev.copy()

    for _ in range(n_classifiers):
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42,
                                 solver='saga', n_jobs=1)
        clf.fit(X_tr, Y_train)
        acc = clf.score(X_dv, Y_dev)

        if acc < min_accuracy:
            break

        W = clf.coef_  # (1, d)

        if first_dir is None:
            # Orient so that harmful samples (label 1) score positively
            scores = X_train @ W.T
            if np.mean(scores[Y_train == 1]) < np.mean(scores[Y_train == 0]):
                W = -W
            norm = np.linalg.norm(W)
            first_dir = W / norm if norm > 1e-9 else W.copy()

        accuracies.append(acc)
        P_rowspace = _get_rowspace_projection(W)
        rowspace_projections.append(P_rowspace)

        # Project onto intersection of current nullspaces (Ben-Israel stability)
        P_current = _get_nullspace_projection(rowspace_projections, d)
        X_tr = P_current.dot(X_train.T).T
        X_dv = P_current.dot(X_dev.T).T

    P = (_get_nullspace_projection(rowspace_projections, d)
         if rowspace_projections else np.eye(d))
    return P, first_dir, accuracies


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
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
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
    n_classifiers: int = 1,
    min_accuracy: float = 0.55,
    val_frac: float = 0.2,
    batch_size: int = 32,
) -> Float[Tensor, "n_positions n_layers d_model"]:
    """Extract INLP refusal directions for all (position, layer) pairs.

    Runs INLP per (pos, layer) to obtain the first (most discriminative)
    refusal direction.  Returns a tensor of the same shape as ``mean_diffs``
    so it can be passed directly to ``select_direction()``.

    Activations are saved to disk so that ``compute_inlp_nullspace_projection``
    can re-use them without re-running the model.

    Parameters
    ----------
    n_classifiers : int
        Maximum INLP iterations per (pos, layer).  1 suffices to obtain the
        first discriminative direction; use a larger value for the nullspace P.
    min_accuracy : float
        Stop INLP early if the classifier accuracy drops below this threshold.

    Returns
    -------
    inlp_directions : Tensor  (n_positions, n_layers, d_model)
    """
    os.makedirs(artifact_dir, exist_ok=True)

    positions = list(range(-len(model_base.eoi_toks), 0))

    # ── Extract activations (one forward pass each) ────────────────────────
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

    # Save for later reuse in compute_inlp_nullspace_projection
    torch.save(harmful_acts,  os.path.join(artifact_dir, "harmful_activations.pt"))
    torch.save(harmless_acts, os.path.join(artifact_dir, "harmless_activations.pt"))

    n_pos, n_layers = len(positions), model_base.model.config.num_hidden_layers
    d_model = model_base.model.config.hidden_size

    # Balance classes
    n_min = min(harmful_acts.shape[0], harmless_acts.shape[0])
    harmful_acts  = harmful_acts[:n_min]
    harmless_acts = harmless_acts[:n_min]

    # Labels: 1 = harmful, 0 = harmless
    Y = np.array([1] * n_min + [0] * n_min, dtype=np.int32)

    n_val = max(4, int(2 * n_min * val_frac))
    rng   = np.random.default_rng(seed=42)
    idx   = rng.permutation(2 * n_min)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    inlp_directions = torch.zeros((n_pos, n_layers, d_model), dtype=torch.float32)

    # ── Run INLP per (pos, layer) ─────────────────────────────────────────
    for pos_idx in range(n_pos):
        src_pos = pos_idx - n_pos   # convert to negative index for logging
        for layer_idx in tqdm(range(n_layers),
                               desc=f"INLP directions (pos {src_pos})"):

            X = torch.cat([
                harmful_acts[:,  pos_idx, layer_idx, :],
                harmless_acts[:, pos_idx, layer_idx, :],
            ], dim=0).numpy().astype(np.float64)   # (2*n_min, d_model)

            _, first_dir, _ = _run_inlp_sklearn(
                X[train_idx], Y[train_idx],
                X[val_idx],   Y[val_idx],
                n_classifiers=n_classifiers,
                min_accuracy=min_accuracy,
            )

            if first_dir is not None:
                inlp_directions[pos_idx, layer_idx] = (
                    torch.from_numpy(first_dir.squeeze()).float()
                )
            # else: leave as zero vector (no classifier reached min_accuracy)

    assert not inlp_directions.isnan().any(), "NaN in INLP directions"
    torch.save(inlp_directions, os.path.join(artifact_dir, "inlp_first_directions.pt"))

    return inlp_directions


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
    P : np.ndarray, shape (d_model, d_model), dtype float64
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
    X = torch.cat([harmful_acts, harmless_acts], dim=0).numpy().astype(np.float64)

    n_val = max(4, int(2 * n_min * val_frac))
    rng   = np.random.default_rng(seed=42)
    idx   = rng.permutation(2 * n_min)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    P, _, accuracies = _run_inlp_sklearn(
        X[train_idx], Y[train_idx],
        X[val_idx],   Y[val_idx],
        n_classifiers=n_classifiers,
        min_accuracy=min_accuracy,
    )

    print(
        f"Nullspace projection: {len(accuracies)} classifiers removed, "
        f"accuracies: {[f'{a:.3f}' for a in accuracies]}"
    )

    return P
