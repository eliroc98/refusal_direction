from pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str, device: str = 'auto',
                         gpu_memory_utilization: float = 0.9) -> ModelBase:

    if 'qwen' in model_path.lower():
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path, device=device, gpu_memory_utilization=gpu_memory_utilization)
    if 'llama-3' in model_path.lower():
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path, device=device, gpu_memory_utilization=gpu_memory_utilization)
    elif 'llama' in model_path.lower():
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path, device=device, gpu_memory_utilization=gpu_memory_utilization)
    elif 'gemma' in model_path.lower():
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path, device=device, gpu_memory_utilization=gpu_memory_utilization)
    elif 'yi' in model_path.lower():
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path, device=device, gpu_memory_utilization=gpu_memory_utilization)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
