import torch
import itertools
import json

from typing import List, Optional
from datasets import load_dataset

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.nnsight_interventions import LayerIntervention, apply_interventions

def batch_iterator_chat_completions(dataset_instructions, dataset_outputs, tokenize_instructions_fn, batch_size, eoi_toks):
    it_instructions = iter(dataset_instructions)
    it_outputs = iter(dataset_outputs)
    while True:
        instructions_batch = list(itertools.islice(it_instructions, batch_size))
        outputs_batch = list(itertools.islice(it_outputs, batch_size))
        if not instructions_batch or not outputs_batch:
            break
        inputs = tokenize_instructions_fn(instructions=instructions_batch, outputs=outputs_batch)

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0

        for b in range(inputs["input_ids"].shape[0]):
            for i in range(inputs["input_ids"].shape[1]):

                if torch.all(inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks):
                    loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
                    break

                # Llama2 tokenization quirk
                if eoi_toks.shape[0] == 6 and (inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks).sum().item() >= eoi_toks.shape[0] - 2:
                    loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
                    break

        yield inputs, loss_mask

def batch_iterator_custom_completions(completions_file_path: str, tokenize_instructions_fn, batch_size, eoi_toks):
    custom_completions = json.load(open(completions_file_path, 'r'))

    instructions, completions = [], []
    for i in range(len(custom_completions)):
        instructions.append(custom_completions[i]['prompt'])
        completions.append(custom_completions[i]['response'])

    return batch_iterator_chat_completions(instructions, completions, tokenize_instructions_fn, batch_size, eoi_toks)

def batch_iterator_alpaca(tokenize_instructions_fn, batch_size, eoi_toks):
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42)

    instructions, completions = [], []
    for i in range(len(dataset)):
        if dataset[i]['input'].strip() == '':
            instructions.append(dataset[i]['instruction'])
            completions.append(dataset[i]['output'])

    return batch_iterator_chat_completions(instructions, completions, tokenize_instructions_fn, batch_size, eoi_toks)

def batch_iterator_pile(tokenizer, batch_size, max_length):
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True, trust_remote_code=True)

    it_dataset = iter(dataset)
    while True:
        batch = list(itertools.islice(it_dataset, batch_size))
        if not batch:
            break
        inputs = tokenizer([b['text'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0

        yield inputs, loss_mask

def compute_loss_over_dataset(
    model_base: ModelBase,
    batch_iterator,
    n_batches=256,
    interventions: Optional[List[LayerIntervention]] = None,
):
    """Compute cross-entropy loss over a dataset using vLLM-backed nnsight trace.

    Processes each sample individually via tracer.invoke() to get clean
    per-sample full-sequence logits from the lm_head, which is required
    for proper cross-entropy loss computation.
    """
    accumulated_loss = torch.tensor(0, dtype=torch.float64)
    accumulated_n_tokens = torch.tensor(0, dtype=torch.int64)

    batch_idx = 0
    for inputs, loss_mask in batch_iterator:
        if n_batches != -1 and batch_idx >= n_batches:
            break

        input_ids = inputs["input_ids"]  # (batch, seq_len) — padded
        batch_size = input_ids.shape[0]

        # Process each sample individually since sequence lengths may differ.
        # nnsight 0.6.3 cannot push batch results back to the outer frame,
        # so we run one trace per prompt (same pattern as model_base.py).
        for b in range(batch_size):
            prompt = model_base.tokenizer.decode(input_ids[b], skip_special_tokens=False)

            with torch.no_grad():
                with model_base.nnsight_model.trace(prompt, temperature=0.0, top_p=1):
                    if interventions:
                        apply_interventions(model_base, interventions)
                    logit_save = model_base._get_lm_head_proxy().output.save()

            raw_logits = logit_save.value
            if raw_logits.dim() == 3:
                raw_logits = raw_logits.squeeze(0)   # (seq_len, vocab)
            raw_logits = raw_logits.cpu()
            raw_logits = torch.where(torch.isnan(raw_logits), torch.zeros_like(raw_logits), raw_logits)
            raw_logits = raw_logits.clamp(min=-1e4, max=1e4)

            # Align to the padded length from the tokenizer.
            # vLLM may produce a different length if it strips padding tokens.
            # Use the minimum of both to stay safe.
            sample_ids = input_ids[b]                  # (padded_seq_len,)
            sample_mask = loss_mask[b]                  # (padded_seq_len,)
            vllm_len = raw_logits.shape[0]
            pad_len = sample_ids.shape[0]
            L = min(vllm_len, pad_len)

            logits = raw_logits[:L]                     # (L, vocab)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # Shift: predict next token from each position
            log_probs_for_labels = log_probs[:-1].gather(
                dim=-1, index=sample_ids[1:L].unsqueeze(-1)
            ).squeeze(-1)                               # (L-1,)

            # Pad back to L for mask alignment
            log_probs_for_labels = torch.cat([
                log_probs_for_labels,
                torch.zeros(1, dtype=log_probs_for_labels.dtype),
            ])                                          # (L,)

            mask = sample_mask[:L].to(log_probs_for_labels.device)
            accumulated_loss += -(log_probs_for_labels * mask).sum()
            accumulated_n_tokens += mask.sum().to(torch.int64)

        batch_idx += 1

    ce_loss = accumulated_loss / accumulated_n_tokens
    perplexity = torch.exp(ce_loss)

    return ce_loss, perplexity, accumulated_n_tokens

def evaluate_loss(
    model_base: ModelBase,
    interventions: Optional[List[LayerIntervention]] = None,
    batch_size=16,
    n_batches=256,
    max_seq_length=256,
    dataset_labels=["pile", "alpaca", "alpaca_custom_completions"],
    completions_file_path=None,
    intervention_label: str = "",
    # Legacy hook arguments — silently ignored; use interventions= instead.
    fwd_pre_hooks=None,
    fwd_hooks=None,
):
    result = {}
    tag = f" [{intervention_label}]" if intervention_label else ""

    for label in dataset_labels:
        if label == 'pile':
            dataset_iterator = batch_iterator_pile(model_base.tokenizer, batch_size=batch_size, max_length=max_seq_length)
            n = n_batches
        elif label == 'alpaca':
            dataset_iterator = batch_iterator_alpaca(model_base.tokenize_instructions_fn, batch_size=batch_size, eoi_toks=torch.tensor(model_base.eoi_toks))
            n = n_batches
        elif label == 'alpaca_custom_completions':
            assert completions_file_path is not None, "A file path must be passed to load the completions"
            dataset_iterator = batch_iterator_custom_completions(
                completions_file_path=completions_file_path,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                batch_size=batch_size,
                eoi_toks=torch.tensor(model_base.eoi_toks)
            )
            n = -1
        else:
            raise ValueError(f"Unknown dataset label: {label}")

        ce_loss, perplexity, n_tokens = compute_loss_over_dataset(
            model_base, batch_iterator=dataset_iterator, n_batches=n, interventions=interventions)
        print(f"{label.upper()} DATASET{tag}:")
        print(f"CE loss: {ce_loss.item()}, Perplexity: {perplexity.item()}, N tokens: {n_tokens.item()}")

        result[label] = {
            "ce_loss": ce_loss.item(),
            "perplexity": perplexity.item(),
            "n_tokens": n_tokens.item()
        }

    return result
