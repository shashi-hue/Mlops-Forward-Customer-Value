import logging
import os
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime
from pathlib import Path

#constants 
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%d_%m_%y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024

#log file path
root_dir = Path(__file__).parent.parent.absolute()
log_dir_path = os.path.join(root_dir, LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)

#configure log
def configure_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
    
    #set file handler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    #console hadler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)


    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

configure_logger()