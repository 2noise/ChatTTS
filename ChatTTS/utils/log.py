import logging
from pathlib import Path

logger = logging.getLogger(Path(__file__).parent.name)

def set_utils_logger(l: logging.Logger):
    global logger
    logger = l
