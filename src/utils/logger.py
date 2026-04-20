import logging
import sys
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a custom logger that outputs to both the console 
    and a centralized log file.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt='%(asctime)s ; %(levelname)-8s ; %(name)s ; %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / 'logs'
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / 'execution.log'
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger