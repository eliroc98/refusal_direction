import torch

from typing import List, Optional
from datasets import load_dataset
from tqdm.auto import tqdm

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.nnsight_interventions import LayerIntervention, apply_interventions


# ── few-shot prompt builders ──────────────────────────────────────────────────

def _mmlu_format_example(row, include_answer=True):
    choices = row["choices"]
    labels = ["A", "B", "C", "D"]
    q = row["question"].strip()
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    prompt = f"Question: {q}\n{opts}\nAnswer:"
    if include_answer:
        correct_idx = row["answer"]
        prompt += f" {labels[correct_idx]}"
    return prompt


def _arc_format_example(row, include_answer=True):
    choices = row["choices"]["text"]
    labels_raw = row["choices"]["label"]
    labels = []
    for l in labels_raw:
        if l in ("A", "B", "C", "D"):
            labels.append(l)
        elif l.isdigit():
            labels.append(chr(ord("A") + int(l) - 1))
        else:
            labels.append(l)
    q = row["question"].strip()
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    correct_raw = row["answerKey"]
    if correct_raw.isdigit():
        correct_label = chr(ord("A") + int(correct_raw) - 1)
    else:
        correct_label = correct_raw
    prompt = f"Question: {q}\n{opts}\nAnswer:"
    if include_answer:
        prompt += f" {correct_label}"
    return prompt, correct_label, labels


def _truthfulqa_format_example(row, include_answer=True):
    choices = row["mc1_targets"]["choices"]
    labels_flag = row["mc1_targets"]["labels"]
    labels = ["A", "B", "C", "D"][: len(choices)]
    correct_idx = labels_flag.index(1)
    q = row["question"].strip()
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    prompt = f"Question: {q}\n{opts}\nAnswer:"
    if include_answer:
        prompt += f" {labels[correct_idx]}"
    return prompt, labels[correct_idx], labels


# ── batched logit-scoring helper ──────────────────────────────────────────────

def _get_last_logits_batched(
    model_base: ModelBase,
    prompts: List[str],
    interventions: Optional[List[LayerIntervention]] = None,
    batch_size: int = 32,
) -> torch.Tensor:
    """Return last-position logits for every prompt via vLLM-backed nnsight tracing.

    Shape: (n_prompts, vocab_size).  Uses per-invoke tracing to get clean
    per-prompt outputs while still benefiting from vLLM batching.
    """
    all_logits = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]

        with torch.no_grad():
            with model_base.nnsight_model.trace() as tracer:
                saved = []
                for prompt in batch:
                    with tracer.invoke(prompt, temperature=0.0, top_p=1):
                        if interventions:
                            apply_interventions(model_base, interventions)
                        saved.append(model_base._get_lm_head_proxy().output.save())

        batch_logits = []
        for s in saved:
            v = s.value
            if v.dim() == 2:
                batch_logits.append(v[-1, :])      # (vocab,)
            else:
                batch_logits.append(v[0, -1, :])    # (vocab,)
        last = torch.stack(batch_logits, dim=0).cpu()  # (batch, vocab)
        last = torch.where(torch.isnan(last), torch.zeros_like(last), last)
        last = last.clamp(min=-1e4, max=1e4)
        all_logits.append(last)

    return torch.cat(all_logits, dim=0)   # (n_prompts, vocab)


def _tok_id(tokenizer, label: str) -> int:
    """Token id for a choice label (with leading space for SentencePiece)."""
    return tokenizer.encode(f" {label}", add_special_tokens=False)[-1]


# ── per-benchmark evaluators ──────────────────────────────────────────────────

def _evaluate_mmlu(model_base: ModelBase, interventions, n_samples, batch_size):
    dataset = load_dataset("cais/mmlu", "all", split="test")
    if n_samples > 0 and n_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(n_samples))

    val_dataset = load_dataset("cais/mmlu", "all", split="validation")
    val_by_subject: dict = {}
    for row in val_dataset:
        val_by_subject.setdefault(row["subject"], []).append(row)

    prompts, answers = [], []
    for row in tqdm(dataset, desc="MMLU (building prompts)", leave=False):
        subj = row["subject"]
        few_shot_rows = val_by_subject.get(subj, [])[:5]
        few_shot_text = "\n\n".join(_mmlu_format_example(r) for r in few_shot_rows)
        question_text = _mmlu_format_example(row, include_answer=False)
        prompt = (few_shot_text + "\n\n" + question_text) if few_shot_text else question_text
        prompts.append(prompt)
        answers.append(row["answer"])

    all_logits = _get_last_logits_batched(model_base, prompts, interventions, batch_size)

    choice_labels = ["A", "B", "C", "D"]
    tok_ids = [_tok_id(model_base.tokenizer, lbl) for lbl in choice_labels]

    correct = 0
    for i, ans in enumerate(answers):
        scores = [all_logits[i, tid].item() for tid in tok_ids]
        predicted = int(torch.tensor(scores).argmax().item())
        if predicted == ans:
            correct += 1

    return correct / len(answers) if answers else None, len(answers)


def _evaluate_arc(model_base: ModelBase, interventions, n_samples, batch_size):
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    dataset = dataset.filter(lambda x: len(x["choices"]["text"]) == 4)
    if n_samples > 0 and n_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(n_samples))

    train_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    train_dataset = train_dataset.filter(lambda x: len(x["choices"]["text"]) == 4)
    few_shot_rows = list(train_dataset.select(range(min(5, len(train_dataset)))))
    few_shot_parts = [_arc_format_example(r, include_answer=True)[0] for r in few_shot_rows]
    few_shot_text = "\n\n".join(few_shot_parts)

    prompts, correct_labels_list, labels_list = [], [], []
    for row in tqdm(dataset, desc="ARC (building prompts)", leave=False):
        question_part, correct_label, labels = _arc_format_example(row, include_answer=False)
        prompt = (few_shot_text + "\n\n" + question_part) if few_shot_text else question_part
        prompts.append(prompt)
        correct_labels_list.append(correct_label)
        labels_list.append(labels)

    all_logits = _get_last_logits_batched(model_base, prompts, interventions, batch_size)

    correct = 0
    for i, (correct_label, labels) in enumerate(zip(correct_labels_list, labels_list)):
        tok_ids = [_tok_id(model_base.tokenizer, lbl) for lbl in labels]
        scores = [all_logits[i, tid].item() for tid in tok_ids]
        predicted_idx = int(torch.tensor(scores).argmax().item())
        if labels[predicted_idx] == correct_label:
            correct += 1

    return correct / len(prompts) if prompts else None, len(prompts)


def _evaluate_truthfulqa(model_base: ModelBase, interventions, n_samples, batch_size):
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    dataset = dataset.filter(lambda x: len(x["mc1_targets"]["choices"]) >= 4)
    if n_samples > 0 and n_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(n_samples))

    prompts, correct_labels_list, labels_list = [], [], []
    for row in tqdm(dataset, desc="TRUTHFULQA (building prompts)", leave=False):
        question_part, correct_label, labels = _truthfulqa_format_example(row, include_answer=False)
        prompts.append(question_part)
        correct_labels_list.append(correct_label)
        labels_list.append(labels)

    all_logits = _get_last_logits_batched(model_base, prompts, interventions, batch_size)

    correct = 0
    for i, (correct_label, labels) in enumerate(zip(correct_labels_list, labels_list)):
        tok_ids = [_tok_id(model_base.tokenizer, lbl) for lbl in labels]
        scores = [all_logits[i, tid].item() for tid in tok_ids]
        predicted_idx = int(torch.tensor(scores).argmax().item())
        if labels[predicted_idx] == correct_label:
            correct += 1

    return correct / len(prompts) if prompts else None, len(prompts)


# ── main entry point ──────────────────────────────────────────────────────────

def evaluate_benchmarks(
    model_base: ModelBase,
    interventions: Optional[List[LayerIntervention]] = None,
    benchmarks=("mmlu", "arc", "truthfulqa"),
    n_mmlu=500,
    n_arc=-1,
    n_truthfulqa=-1,
    batch_size=32,
    intervention_label="",
    # Legacy hook arguments kept for call-site backward compatibility;
    # if passed they are silently ignored (use interventions= instead).
    fwd_pre_hooks=None,
    fwd_hooks=None,
) -> dict:
    """Evaluate MMLU, ARC-Challenge, and TruthfulQA via batched log-prob scoring.

    Returns:
        {
          "mmlu":       {"accuracy": float | None, "n_examples": int},
          "arc":        {"accuracy": float | None, "n_examples": int},
          "truthfulqa": {"accuracy": float | None, "n_examples": int},
        }
    """
    tag = f" [{intervention_label}]" if intervention_label else ""
    result = {}

    _runners = {
        "mmlu":       (_evaluate_mmlu,       n_mmlu),
        "arc":        (_evaluate_arc,        n_arc),
        "truthfulqa": (_evaluate_truthfulqa, n_truthfulqa),
    }

    for bm in benchmarks:
        fn, n = _runners[bm]
        try:
            acc, n_ex = fn(model_base, interventions, n, batch_size)
            print(f"{bm.upper()} BENCHMARK{tag}: accuracy={acc:.4f}, n_examples={n_ex}")
            result[bm] = {"accuracy": acc, "n_examples": n_ex}
        except Exception as e:
            print(f"{bm.upper()} BENCHMARK{tag}: FAILED — {e}")
            result[bm] = {"accuracy": None, "n_examples": 0, "error": str(e)}

    return result
