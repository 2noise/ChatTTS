import torch
import logging

def select_device(min_memory=2048):
    logger = logging.getLogger(__name__)
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.log(logging.INFO, 'Using MPS (Apple Silicon) as the device.')
    elif torch.cuda.is_available():
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_memory = props.total_memory - torch.cuda.memory_reserved(i)
            available_gpus.append((i, free_memory))
        selected_gpu, max_free_memory = max(available_gpus, key=lambda x: x[1])
        device = torch.device(f'cuda:{selected_gpu}')
        free_memory_mb = max_free_memory / (1024 * 1024)
        if free_memory_mb < min_memory:
            logger.log(logging.WARNING, f'GPU {selected_gpu} has {round(free_memory_mb, 2)} MB memory left.')
            device = torch.device('cpu')
    else:
        logger.log(logging.WARNING, 'No GPU found, use CPU instead')
        device = torch.device('cpu')
    
    return device
