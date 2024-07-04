import torch

from .log import logger


def select_device(min_memory=2047, experimental=False):
    if torch.cuda.is_available():
        selected_gpu = 0
        max_free_memory = -1
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_memory = props.total_memory - torch.cuda.memory_reserved(i)
            if max_free_memory < free_memory:
                selected_gpu = i
                max_free_memory = free_memory
        free_memory_mb = max_free_memory / (1024 * 1024)
        if free_memory_mb < min_memory:
            logger.get_logger().warning(
                f"GPU {selected_gpu} has {round(free_memory_mb, 2)} MB memory left. Switching to CPU."
            )
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{selected_gpu}")
    elif torch.backends.mps.is_available():
        """
        Currently MPS is slower than CPU while needs more memory and core utility,
        so only enable this for experimental use.
        """
        if experimental:
            # For Apple M1/M2 chips with Metal Performance Shaders
            logger.get_logger().warning("experimantal: found apple GPU, using MPS.")
            device = torch.device("mps")
        else:
            logger.get_logger().info("found Apple GPU, but use CPU.")
            device = torch.device("cpu")
    else:
        logger.get_logger().warning("no GPU found, use CPU instead")
        device = torch.device("cpu")

    return device
