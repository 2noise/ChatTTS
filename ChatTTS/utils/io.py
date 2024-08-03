import os
import logging
from typing import Union
from dataclasses import is_dataclass

from .log import logger


def get_latest_modified_file(directory):

    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    if not files:
        logger.get_logger().log(
            logging.WARNING, f"no files found in the directory: {directory}"
        )
        return None
    latest_file = max(files, key=os.path.getmtime)

    return latest_file


def del_all(d: Union[dict, list]):
    if is_dataclass(d):
        for k in list(vars(d).keys()):
            x = getattr(d, k)
            if isinstance(x, dict) or isinstance(x, list) or is_dataclass(x):
                del_all(x)
            del x
            delattr(d, k)
    elif isinstance(d, dict):
        lst = list(d.keys())
        for k in lst:
            x = d.pop(k)
            if isinstance(x, dict) or isinstance(x, list) or is_dataclass(x):
                del_all(x)
            del x
    elif isinstance(d, list):
        while len(d):
            x = d.pop()
            if isinstance(x, dict) or isinstance(x, list) or is_dataclass(x):
                del_all(x)
            del x
    else:
        del d
