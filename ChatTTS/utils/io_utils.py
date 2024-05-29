
import os
import logging

def get_latest_modified_file(directory):
    logger = logging.getLogger(__name__)
    
    files = [os.path.join(directory, f) for f in os.listdir(directory)] 
    if not files:
        logger.log(logging.WARNING, f'No files found in the directory: {directory}')
        return None
    latest_file = max(files, key=os.path.getmtime)

    return latest_file