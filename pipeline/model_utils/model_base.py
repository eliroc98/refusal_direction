from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoConfig
from nnsight.modeling.vllm import VLLM
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float

import torch


class ModelBase(ABC):
    def __init__(self, model_name_or_path: str, device: str = 'auto',
                 gpu_memory_utilization: float = 0.9):
        self.model_name_or_path = model_name_or_path
        self.device = device

        # Load model config (lightweight, no weights).
        self._config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True)

        # Determine tensor-parallel size from device string.
        if device == 'auto':
            tp_size = max(1, torch.cuda.device_count())
        else:
            tp_size = 1

        # Create vLLM-backed nnsight model.
        vllm_kwargs = self._get_vllm_kwargs()
        self.nnsight_model = VLLM(
            model_name_or_path,
            dtype=self._get_dtype_str(),
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dispatch=True,
            trust_remote_code=True,
            **vllm_kwargs,
        )

        # Tokenizer: use the one loaded by vLLM, then apply model-specific config.
        self.tokenizer: AutoTokenizer = self.nnsight_model.tokenizer
        self._configure_tokenizer()

        self.n_layers: int = self._config.num_hidden_layers

        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.format_instruction_fn = self._get_format_instruction_fn()
        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks = self._get_refusal_toks()

    def del_model(self):
        if hasattr(self, 'nnsight_model') and self.nnsight_model is not None:
            del self.nnsight_model
            self.nnsight_model = None

    # ── Abstract methods: dtype / vLLM config ─────────────────────────────────

    @abstractmethod
    def _get_dtype_str(self) -> str:
        """Return dtype string for vLLM, e.g. 'bfloat16' or 'float16'."""
        pass

    def _get_vllm_kwargs(self) -> dict:
        """Override to pass extra kwargs to the VLLM constructor."""
        return {}

    @abstractmethod
    def _configure_tokenizer(self):
        """Apply model-specific tokenizer settings (pad_token, padding_side, etc.)."""
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_format_instruction_fn(self):
        """Return a callable(instruction: str) -> str that formats a single
        instruction into the model's chat template (no tokenization)."""
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        pass

    @abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        pass

    @abstractmethod
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        pass

    # ── Abstract nnsight proxy accessors ──────────────────────────────────────
    # Each concrete model implements these to return the nnsight proxy object
    # for the corresponding module at *layer_idx*.  They must be called from
    # inside an active ``nnsight_model.trace()`` context.

    @abstractmethod
    def _get_block_proxy(self, layer_idx: int):
        """nnsight proxy for the transformer block at layer_idx."""
        pass

    @abstractmethod
    def _get_attn_proxy(self, layer_idx: int):
        """nnsight proxy for the attention module at layer_idx."""
        pass

    @abstractmethod
    def _get_mlp_proxy(self, layer_idx: int):
        """nnsight proxy for the MLP module at layer_idx."""
        pass

    def _get_lm_head_proxy(self):
        """nnsight proxy for the language-model head (full-sequence logits).
        Override if the head is not at ``nnsight_model.lm_head``."""
        return self.nnsight_model.lm_head

    def _get_logits_proxy(self):
        """nnsight proxy for the vLLM logits module (last-position logits
        during prefill).  Use this for scoring; use ``_get_lm_head_proxy``
        when full-sequence logits are needed (e.g. loss evaluation)."""
        return self.nnsight_model.logits

    # ── Generation ────────────────────────────────────────────────────────────

    def _apply_interventions_as_edits(self, interventions):
        """Register interventions as persistent edits on the nnsight model.

        Uses ``model.edit()`` so that the interventions are applied at every
        forward pass (including each autoregressive generation step) without
        needing ``tracer.iter``.
        """
        from pipeline.utils.nnsight_interventions import apply_interventions

        with self.nnsight_model.edit():
            apply_interventions(self, interventions)

    def _ensure_vllm_entrypoint(self):
        """Ensure the underlying vLLM LLM engine is initialized."""
        if self.nnsight_model.vllm_entrypoint is None:
            with self.nnsight_model.trace('warmup', temperature=0.0,
                                          max_tokens=1) as tracer:
                _ = self.nnsight_model.samples.output.clone().save()

    def generate_completions(self, dataset, interventions=None,
                              batch_size=8, max_new_tokens=64):
        """Generate completions using vLLM with nnsight interventions."""
        from vllm import SamplingParams

        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]

        if interventions:
            # With interventions: use nnsight trace (one prompt at a time,
            # since nnsight 0.6.3 can't push batch results back).
            self._apply_interventions_as_edits(interventions)
            try:
                for item in tqdm(dataset, desc="Generating completions"):
                    prompt = self.format_instruction_fn(item['instruction'])
                    result = None
                    with self.nnsight_model.trace(prompt, temperature=0.0,
                                                  max_tokens=max_new_tokens) as tracer:
                        result = tracer.result.save()
                    completions.append({
                        'category': item['category'],
                        'prompt': item['instruction'],
                        'response': result[0] if result else '',
                    })
            finally:
                self.nnsight_model.clear_edits()
        else:
            # Without interventions: use vLLM directly for batched generation.
            self._ensure_vllm_entrypoint()
            params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            prompts = [self.format_instruction_fn(inst) for inst in instructions]

            for i in tqdm(range(0, len(dataset), batch_size), desc="Generating completions"):
                batch_prompts = prompts[i:i + batch_size]
                batch_categories = categories[i:i + batch_size]
                batch_instructions = instructions[i:i + batch_size]

                outputs = self.nnsight_model.vllm_entrypoint.generate(
                    batch_prompts, params)
                for j, out in enumerate(outputs):
                    completions.append({
                        'category': batch_categories[j],
                        'prompt': batch_instructions[j],
                        'response': out.outputs[0].text.strip(),
                    })

        return completions
