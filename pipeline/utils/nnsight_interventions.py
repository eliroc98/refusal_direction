
import torch
import numpy as np

from dataclasses import dataclass, field
from typing import List
from torch import Tensor
from jaxtyping import Float


@dataclass
class LayerIntervention:
    layer_idx: int
    target: str        # 'block_input' | 'attn_output' | 'mlp_output'
    intervention_type: str  # 'ablation' | 'actadd' | 'nullspace' | 'reflection'
    params: dict = field(default_factory=dict)


def apply_interventions(model_base, interventions: List[LayerIntervention]):
    """Apply a list of LayerInterventions inside an active nnsight trace() context.

    Must be called between ``model_base.nnsight_model.trace(...)`` and the end
    of that context manager block.
    """
    for iv in interventions:
        # ── resolve the activation proxy ──────────────────────────────────────
        if iv.target == 'block_input':
            # Residual stream entering the transformer block (first positional arg)
            proxy = model_base._get_block_proxy(iv.layer_idx)
            act = proxy.input[0]
        elif iv.target == 'attn_output':
            # First element of the attention module's output tuple
            proxy = model_base._get_attn_proxy(iv.layer_idx)
            act = proxy.output[0]
        elif iv.target == 'mlp_output':
            # MLP output (single tensor for LLaMA-style models)
            proxy = model_base._get_mlp_proxy(iv.layer_idx)
            act = proxy.output
        else:
            raise ValueError(f"Unknown intervention target: {iv.target!r}")

        # ── compute modified activation ────────────────────────────────────────
        if iv.intervention_type == 'ablation':
            direction = iv.params['direction']
            d_norm = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            d_norm = d_norm.to(act)
            new_act = act - (act @ d_norm).unsqueeze(-1) * d_norm

        elif iv.intervention_type == 'actadd':
            vector = iv.params['vector'].to(act)
            coeff = iv.params['coeff']
            new_act = act + coeff * vector

        elif iv.intervention_type == 'nullspace':
            P = iv.params['P']
            if isinstance(P, np.ndarray):
                P = torch.from_numpy(P)
            P_cast = P.to(dtype=act.dtype, device=act.device)
            new_act = torch.matmul(act, P_cast.T)

        elif iv.intervention_type == 'reflection':
            P = iv.params['P']
            alpha = iv.params['alpha']
            if isinstance(P, np.ndarray):
                P = torch.from_numpy(P)
            P = P.float()
            d = P.shape[0]
            P_alpha = alpha * P + (1.0 - alpha) * torch.eye(d, dtype=torch.float32)
            P_cast = P_alpha.to(dtype=act.dtype, device=act.device)
            new_act = torch.matmul(act, P_cast.T)

        else:
            raise ValueError(f"Unknown intervention_type: {iv.intervention_type!r}")

        # ── write back ─────────────────────────────────────────────────────────
        if iv.target == 'block_input':
            proxy.input[0] = new_act
        elif iv.target == 'attn_output':
            proxy.output[0] = new_act
        elif iv.target == 'mlp_output':
            proxy.output = new_act


# ── Builder helpers ────────────────────────────────────────────────────────────

def make_ablation_interventions(
    layers: list,
    direction: Float[Tensor, 'd_model'],
) -> List[LayerIntervention]:
    """Ablate *direction* from block_input, attn_output, and mlp_output at each layer."""
    interventions = []
    for layer in layers:
        interventions.append(LayerIntervention(
            layer_idx=int(layer), target='block_input',
            intervention_type='ablation', params={'direction': direction},
        ))
        interventions.append(LayerIntervention(
            layer_idx=int(layer), target='attn_output',
            intervention_type='ablation', params={'direction': direction},
        ))
        interventions.append(LayerIntervention(
            layer_idx=int(layer), target='mlp_output',
            intervention_type='ablation', params={'direction': direction},
        ))
    return interventions


def make_actadd_interventions(
    layer: int,
    vector: Float[Tensor, 'd_model'],
    coeff: float,
) -> List[LayerIntervention]:
    """Add coeff * vector to the block_input at a single layer."""
    return [LayerIntervention(
        layer_idx=int(layer), target='block_input',
        intervention_type='actadd', params={'vector': vector, 'coeff': coeff},
    )]


def make_nullspace_interventions(
    layers: list,
    P,
) -> List[LayerIntervention]:
    """Project block_input, attn_output, and mlp_output onto P's nullspace at each layer."""
    interventions = []
    for layer in layers:
        for target in ('block_input', 'attn_output', 'mlp_output'):
            interventions.append(LayerIntervention(
                layer_idx=int(layer), target=target,
                intervention_type='nullspace', params={'P': P},
            ))
    return interventions


def make_reflection_interventions(
    layers: list,
    P,
    alpha: float,
) -> List[LayerIntervention]:
    """Apply counterfactual reflection (P_alpha = alpha*P + (1-alpha)*I) at each layer."""
    interventions = []
    for layer in layers:
        for target in ('block_input', 'attn_output', 'mlp_output'):
            interventions.append(LayerIntervention(
                layer_idx=int(layer), target=target,
                intervention_type='reflection', params={'P': P, 'alpha': alpha},
            ))
    return interventions

