
import torch
import contextlib
import functools
import numpy as np

from typing import List, Tuple, Callable, Union
from jaxtyping import Float
from torch import Tensor

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_all_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
):
    fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks

def get_directional_patching_input_pre_hook(direction: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 
        activation += coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn



# ─── Nullspace projection hooks ───────────────────────────────────────────────

def get_nullspace_projection_input_pre_hook(P: Union[np.ndarray, Tensor]):
    """Pre-hook that projects activations onto the nullspace of learned directions.

    P is a (d_model, d_model) projection matrix satisfying P² = P (idempotent).
    Applying P removes all information in the row-space of the INLP classifiers.
    """
    P_tensor = torch.as_tensor(P, dtype=torch.float32) if isinstance(P, np.ndarray) else P

    def hook_fn(module, input):
        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        P_cast = P_tensor.to(dtype=activation.dtype, device=activation.device)
        activation = torch.matmul(activation, P_cast.T)

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_nullspace_projection_output_hook(P: Union[np.ndarray, Tensor]):
    """Post-hook that projects module outputs onto the nullspace of learned directions."""
    P_tensor = torch.as_tensor(P, dtype=torch.float32) if isinstance(P, np.ndarray) else P

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        P_cast = P_tensor.to(dtype=activation.dtype, device=activation.device)
        activation = torch.matmul(activation, P_cast.T)

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn


def get_all_nullspace_projection_hooks(
    model_base,
    P: Union[np.ndarray, Tensor],
):
    """Create nullspace projection hooks for all layers using the same matrix P.

    Mirrors the structure of get_all_direction_ablation_hooks: applies the
    projection to the block residual-stream input (pre-hook) and to the
    attention and MLP outputs (post-hooks).
    """
    n_layers = model_base.model.config.num_hidden_layers
    fwd_pre_hooks = [
        (model_base.model_block_modules[l], get_nullspace_projection_input_pre_hook(P))
        for l in range(n_layers)
    ]
    fwd_hooks = [
        (model_base.model_attn_modules[l], get_nullspace_projection_output_hook(P))
        for l in range(n_layers)
    ]
    fwd_hooks += [
        (model_base.model_mlp_modules[l], get_nullspace_projection_output_hook(P))
        for l in range(n_layers)
    ]
    return fwd_pre_hooks, fwd_hooks


# ─── Per-layer (component-specific) direction hooks ───────────────────────────

def get_all_direction_ablation_hooks_per_layer(
    model_base,
    candidate_directions: Float[Tensor, "n_pos n_layers d_model"],
    pos: int,
):
    """Create ablation hooks where each layer uses its own direction.

    Unlike get_all_direction_ablation_hooks (which applies the same direction
    everywhere), this function applies candidate_directions[pos, layer] to
    layer ``layer``.  This is the "component-specific" ablation variant.

    Parameters
    ----------
    pos : int
        Position index (negative), selecting which token position's directions
        to use (same pos applied across all layers).
    """
    n_layers = model_base.model.config.num_hidden_layers
    fwd_pre_hooks = [
        (model_base.model_block_modules[l],
         get_direction_ablation_input_pre_hook(direction=candidate_directions[pos, l]))
        for l in range(n_layers)
    ]
    fwd_hooks = [
        (model_base.model_attn_modules[l],
         get_direction_ablation_output_hook(direction=candidate_directions[pos, l]))
        for l in range(n_layers)
    ]
    fwd_hooks += [
        (model_base.model_mlp_modules[l],
         get_direction_ablation_output_hook(direction=candidate_directions[pos, l]))
        for l in range(n_layers)
    ]
    return fwd_pre_hooks, fwd_hooks


def get_all_activation_addition_hooks_per_layer(
    model_base,
    candidate_directions: Float[Tensor, "n_pos n_layers d_model"],
    pos: int,
    coeff: float,
):
    """Create activation-addition hooks where each layer uses its own direction.

    Applies candidate_directions[pos, layer] * coeff to the residual-stream
    input of every layer simultaneously.  This is the "component-specific"
    actadd variant.
    """
    n_layers = model_base.model.config.num_hidden_layers
    fwd_pre_hooks = [
        (model_base.model_block_modules[l],
         get_activation_addition_input_pre_hook(
             vector=candidate_directions[pos, l], coeff=coeff))
        for l in range(n_layers)
    ]
    return fwd_pre_hooks, []