import itertools
import hashlib
import json
import math
import time

import torch
from datasets import load_dataset

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase

def _sample_std(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(finite) == 0:
        return None
    if len(finite) == 1:
        return 0.0
    mean = sum(finite) / len(finite)
    return math.sqrt(sum((v - mean) ** 2 for v in finite) / (len(finite) - 1))


def _text_preview(text, max_chars=160):
    text = " ".join(str(text).split())
    return text[:max_chars]


def _safe_exp(value):
    if value is None or not math.isfinite(float(value)) or value > 709:
        return None
    return math.exp(value)


def batch_iterator_chat_completions(dataset_instructions, dataset_outputs, tokenize_instructions_fn, batch_size, eoi_toks, metadata=None):
    it_instructions = iter(dataset_instructions)
    it_outputs = iter(dataset_outputs)
    it_metadata = iter(metadata) if metadata is not None else None
    while True:
        instructions_batch = list(itertools.islice(it_instructions, batch_size))
        outputs_batch = list(itertools.islice(it_outputs, batch_size))
        if not instructions_batch or not outputs_batch:
            break
        if it_metadata is None:
            metadata_batch = [
                {"index": i, "prompt": instruction, "response": output}
                for i, (instruction, output) in enumerate(zip(instructions_batch, outputs_batch))
            ]
        else:
            metadata_batch = list(itertools.islice(it_metadata, batch_size))
        inputs = tokenize_instructions_fn(instructions=instructions_batch, outputs=outputs_batch)

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0 # loss should not be computed for last token position

        # also mask out all tokens before the eoi token region
        for b in range(inputs["input_ids"].shape[0]):
            for i in range(inputs["input_ids"].shape[1]):

                if torch.all(inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks):
                    loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
                    break

                # normally the above condition works. but the tokenization instruction tokens in Llama2 is not clean, and so we need this hack
                if eoi_toks.shape[0] == 6 and (inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks).sum().item() >= eoi_toks.shape[0] - 2:
                    loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
                    break

        yield inputs, loss_mask, metadata_batch

def batch_iterator_custom_completions(completions_file_path: str, tokenize_instructions_fn, batch_size, eoi_toks):
    """Yields batches from the custom completions."""

    with open(completions_file_path, 'r') as f:
        custom_completions = json.load(f)

    instructions, completions, metadata = [], [], []

    for i in range(len(custom_completions)):
        prompt = custom_completions[i]['prompt']
        response = custom_completions[i]['response']
        instructions.append(prompt)
        completions.append(response)
        metadata.append({
            "index": i,
            "prompt": prompt,
            "response": response,
        })

    return batch_iterator_chat_completions(instructions, completions, tokenize_instructions_fn, batch_size, eoi_toks, metadata)

def batch_iterator_alpaca(tokenize_instructions_fn, batch_size, eoi_toks):
    """Yields batches from the Alpaca dataset."""

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42)

    instructions, completions, metadata = [], [], []

    for i in range(len(dataset)):
        if dataset[i]['input'].strip() == '': # filter for instructions that do not have inputs
            prompt = dataset[i]['instruction']
            response = dataset[i]['output']
            instructions.append(prompt)
            completions.append(response)
            metadata.append({
                "index": len(metadata),
                "dataset_index": i,
                "prompt": prompt,
                "response": response,
            })

    return batch_iterator_chat_completions(instructions, completions, tokenize_instructions_fn, batch_size, eoi_toks, metadata)

def batch_iterator_pile(tokenizer, batch_size, max_length):
    """Yields batches from the Pile dataset."""
    dataset = None
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            print(f"Loading pile-uncopyrighted in streaming mode (attempt {attempt + 1}/{max_retries})...")
            dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True, trust_remote_code=True)
            print("Successfully loaded pile-uncopyrighted")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"All {max_retries} connection attempts failed")
                raise e

    it_dataset = iter(dataset)
    example_idx = 0
    while True:
        batch = list(itertools.islice(it_dataset, batch_size))
        if not batch:
            break
        texts = [b['text'] for b in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0 # loss should not be computed for last token position

        metadata_batch = []
        for text in texts:
            text_str = str(text)
            metadata_batch.append({
                "index": example_idx,
                "text_preview": _text_preview(text_str),
                "text_sha256": hashlib.sha256(text_str.encode("utf-8")).hexdigest(),
            })
            example_idx += 1

        yield inputs, loss_mask, metadata_batch

def compute_loss_over_dataset(model, tokenizer, batch_iterator, n_batches=256, fwd_pre_hooks=[], fwd_hooks=[]):
    accumulated_loss = torch.tensor(0, dtype=torch.float64, device=model.device)
    accumulated_n_tokens = torch.tensor(0, dtype=torch.int64, device=model.device)
    per_example = []

    batch_idx = 0
    for inputs, loss_mask, metadata_batch in batch_iterator:
        if n_batches != -1 and batch_idx >= n_batches:
            break

        inputs = inputs.to(model.device)
        loss_mask = loss_mask.to(model.device)

        input_ids = inputs["input_ids"]

        with torch.no_grad(), add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            model_outputs = model(**inputs)

        logits = model_outputs.logits
        # Per-layer ablation can destabilize intermediate activations, producing
        # NaN/inf logits.  Replace NaN with 0 (→ uniform prob) and clamp inf so
        # log_softmax produces finite values.  The resulting high loss correctly
        # reflects a broken model.
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        logits = logits.clamp(min=-1e4, max=1e4)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_for_labels = log_probs[:, :-1].gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # add a last column of zeros to log_probs_for_labels to match the shape of loss_mask
        log_probs_for_labels = torch.cat(
            [
                log_probs_for_labels,
                torch.zeros(log_probs_for_labels.shape[0]).unsqueeze(-1).to(log_probs_for_labels)
            ],
            dim=-1
        )

        # apply loss_mask
        log_probs_for_labels = log_probs_for_labels * loss_mask.to(log_probs_for_labels.device)

        per_example_loss = -log_probs_for_labels.sum(dim=1).detach().to(torch.float64)
        per_example_n_tokens = loss_mask.sum(dim=1).detach().to(torch.int64)

        accumulated_loss += per_example_loss.sum()
        accumulated_n_tokens += per_example_n_tokens.sum()

        for row_idx in range(per_example_loss.shape[0]):
            n_tokens = int(per_example_n_tokens[row_idx].item())
            loss_sum = float(per_example_loss[row_idx].item())
            if n_tokens > 0:
                ce_loss = loss_sum / n_tokens
                perplexity = _safe_exp(ce_loss)
            else:
                ce_loss = None
                perplexity = None

            metadata = dict(metadata_batch[row_idx]) if row_idx < len(metadata_batch) else {}
            metadata.update({
                "n_tokens": n_tokens,
                "ce_loss": ce_loss,
                "perplexity": perplexity,
            })
            per_example.append(metadata)

        batch_idx += 1

    ce_loss = accumulated_loss / accumulated_n_tokens
    ce_loss_value = ce_loss.item()
    perplexities = [row.get("perplexity") for row in per_example]

    return {
        "ce_loss": ce_loss_value,
        "perplexity": _safe_exp(ce_loss_value),
        "perplexity_std": _sample_std(perplexities),
        "n_tokens": accumulated_n_tokens.item(),
        "n_examples": len(per_example),
        "per_example": per_example,
    }

def evaluate_loss(
    model_base: ModelBase,
    fwd_pre_hooks=[],
    fwd_hooks=[],
    batch_size=16,
    n_batches=256,
    max_seq_length=256,
    dataset_labels=["pile", "alpaca", "alpaca_custom_completions"],
    completions_file_path=None,
    custom_completions_file_paths=None,
    intervention_label: str = "",
):
    result = {}
    custom_completions_file_paths = custom_completions_file_paths or {}

    tag = f" [{intervention_label}]" if intervention_label else ""

    for label in dataset_labels:
        if label == 'pile':
            dataset_iterator = batch_iterator_pile(model_base.tokenizer, batch_size=batch_size, max_length=max_seq_length)
            n = n_batches
        elif label == 'alpaca':
            dataset_iterator = batch_iterator_alpaca(model_base.tokenize_instructions_fn, batch_size=batch_size, eoi_toks=torch.tensor(model_base.eoi_toks))
            n = n_batches
        elif label == 'alpaca_custom_completions' or label in custom_completions_file_paths:
            path = custom_completions_file_paths.get(label, completions_file_path)
            assert path is not None, "A file path must be passed to load the completions"

            dataset_iterator = batch_iterator_custom_completions(
                completions_file_path=path,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                batch_size=batch_size,
                eoi_toks=torch.tensor(model_base.eoi_toks)
            )
            n = -1 # process all completions
        else:
            raise ValueError(f"Unknown dataset label: {label}")

        loss_stats = compute_loss_over_dataset(model_base.model, model_base.tokenizer, dataset_iterator, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, n_batches=n)
        print(f"{label.upper()} DATASET{tag}:")
        print(
            f"CE loss: {loss_stats['ce_loss']}, "
            f"Perplexity: {loss_stats['perplexity']}, "
            f"Perplexity std: {loss_stats['perplexity_std']}, "
            f"N tokens: {loss_stats['n_tokens']}, "
            f"N examples: {loss_stats['n_examples']}"
        )

        result[label] = loss_stats

    return result
