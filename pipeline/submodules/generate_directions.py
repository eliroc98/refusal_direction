import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.model_utils.model_base import ModelBase

def get_mean_activations(model_base: ModelBase, instructions, tokenize_instructions_fn, batch_size=32, positions=[-1]):
    """Extract mean block-input activations using nnsight vLLM tracing.

    For each layer, saves the block input activations at the specified positions
    and accumulates the mean across all instructions.
    """
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model_base.n_layers
    n_samples = len(instructions)
    d_model = model_base._config.hidden_size

    # High-precision accumulator to avoid numerical issues.
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64)

    for i in tqdm(range(0, len(instructions), batch_size)):
        batch = instructions[i:i+batch_size]
        prompts = [model_base.format_instruction_fn(inst) for inst in batch]

        with torch.no_grad():
            with model_base.nnsight_model.trace() as tracer:
                # One invoke per sample for clean per-sample outputs.
                per_sample_acts = []
                for prompt in prompts:
                    with tracer.invoke(prompt, temperature=0.0, top_p=1):
                        sample_acts = []
                        for layer in range(n_layers):
                            act = model_base._get_block_proxy(layer).input[0].save()
                            sample_acts.append(act)
                        per_sample_acts.append(sample_acts)

        # Accumulate: per_sample_acts[s][l].value is (1, seq_len_s, d_model) or (seq_len_s, d_model)
        for s_idx, sample_acts in enumerate(per_sample_acts):
            for layer in range(n_layers):
                act = sample_acts[layer].value
                if act.dim() == 3:
                    act = act.squeeze(0)            # (seq_len, d_model)
                # positions are negative indices (e.g. [-2, -1])
                act_at_pos = act[positions, :]      # (n_positions, d_model)
                mean_activations[:, layer] += (1.0 / n_samples) * act_at_pos.to(mean_activations)

    return mean_activations

def get_mean_diff(model_base: ModelBase, harmful_instructions, harmless_instructions, tokenize_instructions_fn, batch_size=32, positions=[-1]):
    mean_activations_harmful = get_mean_activations(model_base, harmful_instructions, tokenize_instructions_fn, batch_size=batch_size, positions=positions)
    mean_activations_harmless = get_mean_activations(model_base, harmless_instructions, tokenize_instructions_fn, batch_size=batch_size, positions=positions)

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_harmful - mean_activations_harmless

    return mean_diff

def generate_directions(model_base: ModelBase, harmful_instructions, harmless_instructions):

    mean_diffs = get_mean_diff(model_base, harmful_instructions, harmless_instructions, model_base.tokenize_instructions_fn, positions=list(range(-len(model_base.eoi_toks), 0)))

    assert mean_diffs.shape == (len(model_base.eoi_toks), model_base.n_layers, model_base._config.hidden_size)
    assert not mean_diffs.isnan().any()

    return mean_diffs
