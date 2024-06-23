
import torch

from .log import logger

def select_device(min_memory=2048):
    if torch.cuda.is_available():
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_memory = props.total_memory - torch.cuda.memory_reserved(i)
            available_gpus.append((i, free_memory))
        selected_gpu, max_free_memory = max(available_gpus, key=lambda x: x[1])
        device = torch.device(f'cuda:{selected_gpu}')
        free_memory_mb = max_free_memory / (1024 * 1024)
        if free_memory_mb < min_memory:
            logger.get_logger().warning(f'GPU {selected_gpu} has {round(free_memory_mb, 2)} MB memory left. Switching to CPU.')
            device = torch.device('cpu')
    elif torch.backends.mps.is_available():
        # For Apple M1/M2 chips with Metal Performance Shaders
        logger.get_logger().info('Apple GPU found, using MPS.')
        device = torch.device('mps')
    else:
        logger.get_logger().warning('No GPU found, use CPU instead')
        device = torch.device('cpu')

    return device
