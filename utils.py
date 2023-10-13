import logging
from lassy import update_progress_bar

logging.basicConfig(level=logging.INFO)

def update_progress(progress):
    logging.info(f"Progress: {progress}")
    update_progress_bar(progress)