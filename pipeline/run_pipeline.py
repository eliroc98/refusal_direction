import gc
import torch
import random
import json
import os
import argparse

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_all_direction_ablation_hooks,
    get_nullspace_projection_input_pre_hook,
)

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.generate_directions_inlp import (
    generate_directions_inlp,
    select_direction_inlp,
)
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for model loading. Use "auto" to spread across all available GPUs, '
                             'or specify a single device such as "cuda:0" or "cpu". (default: auto)')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.9,
                        help='Fraction of GPU memory vLLM classifiers (LlamaGuard2, HarmBench) may use. '
                             'Lower this if the main model and classifiers compete for memory. (default: 0.9)')
    parser.add_argument('--resume_from_eval', action='store_true',
                        help='Skip model inference (steps 1-5) and resume from LlamaGuard evaluation. '
                             'Assumes completions have already been generated in a previous run.')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Run only the inference steps (1-5) and skip the LlamaGuard evaluation. '
                             'Use --resume_from_eval in a separate process to run evaluation afterwards, '
                             'freeing GPU memory between the two phases.')
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

    return harmful_train, harmless_train, harmful_val, harmless_val

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save mean-difference candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs

def generate_and_save_inlp_directions(cfg, model_base, harmful_train, harmless_train):
    """Cache activations to disk for later INLP direction finding."""
    artifact_dir = os.path.join(cfg.artifact_path(), 'generate_directions_inlp')
    generate_directions_inlp(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=artifact_dir,
    )

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, label=''):
    """Select and save the best direction using the standard selection criteria.

    Parameters
    ----------
    label : str
        Optional suffix for artifact filenames (e.g. 'inlp') to keep results
        from different direction-extraction methods separate.
    """
    suffix = f'_{label}' if label else ''
    artifact_dir = os.path.join(cfg.artifact_path(), f'select_direction{suffix}')
    os.makedirs(artifact_dir, exist_ok=True)

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=artifact_dir,
    )

    with open(f'{cfg.artifact_path()}/direction_metadata{suffix}.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction{suffix}.pt')

    return pos, layer, direction

def select_and_save_direction_inlp(cfg, model_base, harmful_val, harmless_val, n_classifiers=20):
    """Select INLP direction and nullspace projection using nullspace-projection scoring.

    Runs full INLP for every (pos, layer) pair, scores each by how much its
    nullspace projection P reduces refusal on harmful_val, and picks the best.
    The first INLP classifier direction from that run is saved as the INLP
    direction for downstream activation-addition analysis.
    """
    import numpy as np
    artifact_dir = os.path.join(cfg.artifact_path(), 'generate_directions_inlp')

    pos, layer, first_dir, P = select_direction_inlp(
        artifact_dir=artifact_dir,
        model_base=model_base,
        harmful_instructions=harmful_val,
        harmless_instructions=harmless_val,
        n_classifiers=n_classifiers,
        min_accuracy=0.55,
    )

    direction = torch.from_numpy(first_dir).float()

    with open(f'{cfg.artifact_path()}/direction_metadata_inlp.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction_inlp.pt')
    np.save(os.path.join(cfg.artifact_path(), 'nullspace_projection.npy'), P)

    return pos, layer, direction, P


def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies, llamaguard2_classifier=None):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        llamaguard2_classifier=llamaguard2_classifier,
    )

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path, intervention_label=intervention_label)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

def _run_inference(cfg, model_path):
    """Steps 1-5: model loading, direction extraction, completions, and loss evaluation."""
    model_base = construct_model_base(cfg.model_path, device=cfg.device)

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)

    # 1. Generate candidate refusal directions
    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)

    # 1b. Generate candidate directions via INLP (discriminatively trained)
    generate_and_save_inlp_directions(cfg, model_base, harmful_train, harmless_train)

    # 2. Select the most effective mean-difference direction
    pos, layer, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions)

    # 2b+c. Select INLP direction using nullspace-projection scoring.
    #       For each (pos, layer), runs full INLP -> P, applies P to harmful_val,
    #       and picks the (pos, layer) where P most reduces refusal (highest
    #       steering score).  The first INLP classifier direction from that run
    #       is used as the direction for downstream activation-addition analysis.
    inlp_pos, inlp_layer, inlp_direction, P = select_and_save_direction_inlp(
        cfg, model_base, harmful_val, harmless_val, n_classifiers=20
    )
    #TODO: check if direction is not flipped

    # -- Build intervention hooks --------------------------------------------------

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []

    # Mean-difference direction: ablate across all layers / actadd at selected layer
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

    # INLP nullspace projection: applied only at the layer from which P was extracted
    nullspace_fwd_pre_hooks, nullspace_fwd_hooks = [
        (model_base.model_block_modules[inlp_layer],
         get_nullspace_projection_input_pre_hook(P))
    ], []
    
    #TODO: ablate INLP direction

    # INLP first direction: actadd at the selected INLP layer (coeff=-1 to jailbreak)
    inlp_actadd_fwd_pre_hooks, inlp_actadd_fwd_hooks = [
        (model_base.model_block_modules[inlp_layer],
         get_activation_addition_input_pre_hook(vector=inlp_direction, coeff=-1.0))
    ], []

    # 3a. Generate and save completions on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', dataset_name)

        # Nullspace projection intervention (applied at the INLP source layer only)
        generate_and_save_completions_for_dataset(cfg, model_base, nullspace_fwd_pre_hooks, nullspace_fwd_hooks, 'nullspace', dataset_name)

        # INLP first-direction actadd (discriminatively trained direction)
        generate_and_save_completions_for_dataset(cfg, model_base, inlp_actadd_fwd_pre_hooks, inlp_actadd_fwd_hooks, 'inlp_actadd', dataset_name)

    # 4a. Generate and save completions on harmless evaluation dataset
    #     (We test whether each intervention INDUCES refusal on harmless prompts,
    #      which would indicate the intervention is not perfectly specific to harmful
    #      instructions.  Ablation-style hooks are omitted here because removing the
    #      refusal direction on already-harmless prompts is not a refusal-induction test.)
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)

    # Mean-diff actadd with coeff=+1.0: add refusal direction to check induction
    actadd_refusal_pre_hooks, actadd_refusal_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))], []
    generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, 'actadd', 'harmless', dataset=harmless_test)

    # INLP actadd with coeff=+1.0: add INLP direction to check refusal induction
    inlp_actadd_refusal_pre_hooks, inlp_actadd_refusal_hooks = [
        (model_base.model_block_modules[inlp_layer],
         get_activation_addition_input_pre_hook(vector=inlp_direction, coeff=+1.0))
    ], []
    generate_and_save_completions_for_dataset(cfg, model_base, inlp_actadd_refusal_pre_hooks, inlp_actadd_refusal_hooks, 'inlp_actadd', 'harmless', dataset=harmless_test)

    # 5. Evaluate loss on harmless datasets for all interventions
    #    (checks whether interventions degrade performance on benign prompts)
    evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
    evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')
    evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd')

    # Nullspace projection loss (should have low impact on harmless perplexity)
    evaluate_loss_for_datasets(cfg, model_base, nullspace_fwd_pre_hooks, nullspace_fwd_hooks, 'nullspace')

    # INLP actadd loss
    evaluate_loss_for_datasets(cfg, model_base, inlp_actadd_fwd_pre_hooks, inlp_actadd_fwd_hooks, 'inlp_actadd')

    # Free ALL GPU resources before loading LlamaGuard2 for evaluation.
    model_base.del_model()
    del model_base
    del candidate_directions, direction, inlp_direction, P
    del baseline_fwd_pre_hooks, baseline_fwd_hooks
    del ablation_fwd_pre_hooks, ablation_fwd_hooks
    del actadd_fwd_pre_hooks, actadd_fwd_hooks
    del nullspace_fwd_pre_hooks, nullspace_fwd_hooks
    del inlp_actadd_fwd_pre_hooks, inlp_actadd_fwd_hooks
    del actadd_refusal_pre_hooks, actadd_refusal_hooks
    del inlp_actadd_refusal_pre_hooks, inlp_actadd_refusal_hooks

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_evaluation(cfg):
    """Steps 3b/4b: LlamaGuard evaluation on previously generated completions."""
    # Pin vllm to the same GPU used by the main model so it doesn't default to GPU 0.
    if cfg.device not in ('auto', 'cpu'):
        gpu_index = cfg.device.split(':')[-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index

    # Create the LlamaGuard2 classifier once and reuse it for all evaluations
    # to avoid repeated 15GB model loading/unloading and OOM from memory fragmentation.
    from pipeline.submodules.evaluate_jailbreak import LlamaGuard2Classifier
    lg2_classifier = None
    if "llamaguard2" in cfg.jailbreak_eval_methodologies:
        lg2_classifier = LlamaGuard2Classifier(gpu_memory_utilization=cfg.vllm_gpu_memory_utilization)

    # 3b. Evaluate completions and save results on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, llamaguard2_classifier=lg2_classifier)
        evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, llamaguard2_classifier=lg2_classifier)
        evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, llamaguard2_classifier=lg2_classifier)

        # Nullspace projection evaluation
        evaluate_completions_and_save_results_for_dataset(cfg, 'nullspace', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, llamaguard2_classifier=lg2_classifier)

        # INLP actadd evaluation
        evaluate_completions_and_save_results_for_dataset(cfg, 'inlp_actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, llamaguard2_classifier=lg2_classifier)

    # 4b. Evaluate completions and save results on harmless evaluation dataset
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)

    # INLP actadd harmless evaluation
    evaluate_completions_and_save_results_for_dataset(cfg, 'inlp_actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)

    # Clean up the LlamaGuard2 classifier
    if lg2_classifier is not None:
        lg2_classifier.cleanup()


def run_pipeline(model_path, device='auto', vllm_gpu_memory_utilization=0.9,
                 resume_from_eval=False, skip_eval=False):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, device=device,
                 vllm_gpu_memory_utilization=vllm_gpu_memory_utilization)

    if not resume_from_eval:
        _run_inference(cfg, model_path)

    if not skip_eval:
        _run_evaluation(cfg)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, device=args.device,
                 vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                 resume_from_eval=args.resume_from_eval,
                 skip_eval=args.skip_eval)
