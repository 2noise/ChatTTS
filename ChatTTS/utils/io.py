
import os
import logging
from typing import Union

def get_latest_modified_file(directory):
    logger = logging.getLogger(__name__)
    
    files = [os.path.join(directory, f) for f in os.listdir(directory)] 
    if not files:
        logger.log(logging.WARNING, f'No files found in the directory: {directory}')
        return None
    latest_file = max(files, key=os.path.getmtime)

    return latest_file

def del_all(d: Union[dict, list]):
    if isinstance(d, dict):
        lst = list(d.keys())
        for k in lst:
            x = d.pop(k)
            if isinstance(x, dict) or isinstance(x, list):
                del_all(x)
            del x
        return
    elif isinstance(d, list):
        while len(d):
            x = d.pop()
            if isinstance(x, dict) or isinstance(x, list):
                del_all(x)
            del x
        return
