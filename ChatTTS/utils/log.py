import logging
from pathlib import Path


class Logger:
    def __init__(self, logger=logging.getLogger(Path(__file__).parent.name)):
        self.logger = logger

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def get_logger(self) -> logging.Logger:
        return self.logger


logger = Logger()
