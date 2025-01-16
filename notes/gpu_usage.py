from enum import Enum
from huggingface_hub import get_safetensors_metadata


class BytesUsage(float, Enum):
    INT4: float = 0.5
    INT8: float = 1.0
    FP8: float = 1.0
    FLOAT16: float = 2.0
    BFLOAT16: float = 2.0
    FLOAT32: float = 4


def calculate_memory_usage(num_parameters: float,
                           bytes_usage: BytesUsage) -> float:
    """
    Calculate memory required for serving a model
    Below is the formula:
        MEM = 1.20 * (num_parameters * bytes_usage) / (1024**3)
        1.20 represents ~20% overhead for additional GPU memory requirements

    Args:
        num_parameters: Number of model parameters.
        bytes_usage: data type to represent parameter.

    Returns:
        Estimated GPU memory required in Gigabytes


    """

    memory = round((num_parameters * bytes_usage) / (1024 ** 3), 2)

    return memory


def estimate_memory(model_id: str, dtype: BytesUsage) -> float:
    """
    Estimate memory required for serving a model from huggingface
    Args:
        model_id: Model id from huggingface
        dtype: data type to represent parameter.

    Returns:
        Memory required to serve model
    """
    metadata = get_safetensors_metadata(model_id)
    if not metadata or not metadata.parameter_count:
        raise ValueError(f"Could not find metadata in {model_id}")
    num_parameters = list(metadata.parameter_count.values())[0]

    memory = calculate_memory_usage(num_parameters, dtype)
    print(f"Memory required for serving {model_id} is {memory} GB")
    return memory


if __name__ == '__main__':
    model_name = "clapAI/modernBERT-base-multilingual-sentiment"
    estimate_memory(model_name, BytesUsage.FLOAT16)
