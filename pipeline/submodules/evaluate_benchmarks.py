import math

import torch

from datasets import load_dataset
from tqdm.auto import tqdm

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase


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
    # Normalise digit labels ("1"→"A", etc.)
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
    labels_flag = row["mc1_targets"]["labels"]  # 1 at the correct index
    labels = ["A", "B", "C", "D"][: len(choices)]
    correct_idx = labels_flag.index(1)
    q = row["question"].strip()
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    prompt = f"Question: {q}\n{opts}\nAnswer:"
    if include_answer:
        prompt += f" {labels[correct_idx]}"
    return prompt, labels[correct_idx], labels


# ── logit-scoring helper ──────────────────────────────────────────────────────

def _sample_std(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(finite) == 0:
        return None
    if len(finite) == 1:
        return 0.0
    mean = sum(finite) / len(finite)
    return math.sqrt(sum((v - mean) ** 2 for v in finite) / (len(finite) - 1))


def _accuracy_summary(records):
    n_examples = len(records)
    if n_examples == 0:
        return {
            "accuracy": None,
            "accuracy_std": None,
            "n_examples": 0,
            "per_example": [],
        }
    correctness = [1.0 if r["is_correct"] else 0.0 for r in records]
    return {
        "accuracy": sum(correctness) / n_examples,
        "accuracy_std": _sample_std(correctness),
        "n_examples": n_examples,
        "per_example": records,
    }


def _score_choices(model, tokenizer, prompt, choice_labels, device, fwd_pre_hooks, fwd_hooks):
    """Return index of highest-logit choice label plus per-choice scores."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad(), add_hooks(
        module_forward_pre_hooks=fwd_pre_hooks,
        module_forward_hooks=fwd_hooks,
    ):
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # (vocab,)
    logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
    logits = logits.clamp(min=-1e4, max=1e4)

    scores = []
    for lbl in choice_labels:
        # Leading-space variant for SentencePiece tokenisers
        tok_id = tokenizer.encode(f" {lbl}", add_special_tokens=False)[-1]
        scores.append(logits[tok_id].item())
    predicted_idx = int(torch.tensor(scores).argmax().item())
    return predicted_idx, {lbl: score for lbl, score in zip(choice_labels, scores)}


# ── per-benchmark evaluators ──────────────────────────────────────────────────

def _evaluate_mmlu(model_base: ModelBase, fwd_pre_hooks, fwd_hooks, n_samples):
    model     = model_base.model
    tokenizer = model_base.tokenizer
    device    = model.device

    dataset = load_dataset("cais/mmlu", "all", split="test")
    if n_samples > 0 and n_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(n_samples))

    # Build 5-shot examples from the validation split (first 5 from the same subject)
    val_dataset = load_dataset("cais/mmlu", "all", split="validation")
    val_by_subject: dict = {}
    for row in val_dataset:
        subj = row["subject"]
        val_by_subject.setdefault(subj, []).append(row)

    records = []
    labels = ["A", "B", "C", "D"]
    for idx, row in enumerate(tqdm(dataset, desc="MMLU", leave=False)):
        subj = row["subject"]
        few_shot_rows = val_by_subject.get(subj, [])[:5]
        few_shot_text = "\n\n".join(_mmlu_format_example(r) for r in few_shot_rows)
        question_text = _mmlu_format_example(row, include_answer=False)
        prompt = (few_shot_text + "\n\n" + question_text) if few_shot_text else question_text

        predicted, choice_scores = _score_choices(model, tokenizer, prompt, labels, device, fwd_pre_hooks, fwd_hooks)
        correct_label = labels[row["answer"]]
        predicted_label = labels[predicted]
        records.append({
            "index": idx,
            "subject": subj,
            "question": row["question"],
            "choices": {label: choice for label, choice in zip(labels, row["choices"])},
            "correct_answer": correct_label,
            "predicted_answer": predicted_label,
            "choice_scores": choice_scores,
            "is_correct": predicted_label == correct_label,
        })

    return _accuracy_summary(records)


def _evaluate_arc(model_base: ModelBase, fwd_pre_hooks, fwd_hooks, n_samples):
    model     = model_base.model
    tokenizer = model_base.tokenizer
    device    = model.device

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    # Filter to examples with exactly 4 choices
    dataset = dataset.filter(lambda x: len(x["choices"]["text"]) == 4)
    if n_samples > 0 and n_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(n_samples))

    # 5-shot from train split
    train_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    train_dataset = train_dataset.filter(lambda x: len(x["choices"]["text"]) == 4)
    few_shot_rows = list(train_dataset.select(range(min(5, len(train_dataset)))))
    few_shot_parts = []
    for r in few_shot_rows:
        part, _, _ = _arc_format_example(r, include_answer=True)
        few_shot_parts.append(part)
    few_shot_text = "\n\n".join(few_shot_parts)

    records = []
    for idx, row in enumerate(tqdm(dataset, desc="ARC", leave=False)):
        _, correct_label, labels = _arc_format_example(row, include_answer=False)
        question_part, _, _ = _arc_format_example(row, include_answer=False)
        prompt = (few_shot_text + "\n\n" + question_part) if few_shot_text else question_part

        predicted_idx, choice_scores = _score_choices(model, tokenizer, prompt, labels, device, fwd_pre_hooks, fwd_hooks)
        predicted_label = labels[predicted_idx]
        records.append({
            "index": idx,
            "id": row.get("id"),
            "question": row["question"],
            "choices": {label: choice for label, choice in zip(labels, row["choices"]["text"])},
            "correct_answer": correct_label,
            "predicted_answer": predicted_label,
            "choice_scores": choice_scores,
            "is_correct": predicted_label == correct_label,
        })

    return _accuracy_summary(records)


def _evaluate_truthfulqa(model_base: ModelBase, fwd_pre_hooks, fwd_hooks, n_samples):
    if n_samples == 0:
        return {"accuracy": None, "accuracy_std": None, "n_examples": 0, "per_example": []}
    model     = model_base.model
    tokenizer = model_base.tokenizer
    device    = model.device

    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    # Filter to examples with >= 4 mc1 choices
    dataset = dataset.filter(lambda x: len(x["mc1_targets"]["choices"]) >= 4)
    if n_samples > 0 and n_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(n_samples))

    correct = 0
    total   = 0
    for row in tqdm(dataset, desc="TRUTHFULQA", leave=False):
        _, correct_label, labels = _truthfulqa_format_example(row, include_answer=False)
        question_part, _, _ = _truthfulqa_format_example(row, include_answer=False)
        # 0-shot (standard for TruthfulQA)
        predicted_idx, _ = _score_choices(model, tokenizer, question_part, labels, device, fwd_pre_hooks, fwd_hooks)
        if labels[predicted_idx] == correct_label:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else None
    return {
        "accuracy": acc,
        "accuracy_std": None,
        "n_examples": total,
        "per_example": [],
    }


# ── main entry point ──────────────────────────────────────────────────────────

def evaluate_benchmarks(
    model_base: ModelBase,
    fwd_pre_hooks=[],
    fwd_hooks=[],
    benchmarks=("mmlu", "arc", "truthfulqa"),
    n_mmlu=500,
    n_arc=-1,
    n_truthfulqa=-1,
    intervention_label="",
) -> dict:
    """
    Evaluate MMLU, ARC-Challenge, and TruthfulQA via log-prob scoring.

    Returns:
        {
          "mmlu":       {"accuracy": float | None, "accuracy_std": float | None, "n_examples": int, "per_example": list},
          "arc":        {"accuracy": float | None, "accuracy_std": float | None, "n_examples": int, "per_example": list},
          "truthfulqa": {"accuracy": float | None, "accuracy_std": float | None, "n_examples": int, "per_example": list},
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
            summary = fn(model_base, fwd_pre_hooks, fwd_hooks, n)
            acc = summary.get("accuracy")
            n_ex = summary.get("n_examples", 0)
            if acc is None:
                print(f"{bm.upper()} BENCHMARK{tag}: skipped, n_examples={n_ex}")
            else:
                print(
                    f"{bm.upper()} BENCHMARK{tag}: "
                    f"accuracy={acc:.4f}, accuracy_std={summary.get('accuracy_std')}, "
                    f"n_examples={n_ex}"
                )
            result[bm] = summary
        except Exception as e:
            print(f"{bm.upper()} BENCHMARK{tag}: FAILED — {e}")
            result[bm] = {"accuracy": None, "accuracy_std": None, "n_examples": 0, "per_example": [], "error": str(e)}

    return result
