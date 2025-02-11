import logging
import os

# Define log file path
LOG_FILE = "logs.txt"

# Create a formatter for logs
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Function to get a logger
def get_logger(name: str):
    """
    Returns a configured logger for the given module.
    
    Args:
        name (str): Name of the module requesting the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)  # Unique logger per module
    logger.setLevel(logging.DEBUG)  # Set global logging level

    # Check if handlers already exist (prevents duplicate logs)
    if not logger.handlers:
        # Create file handler (logs stored in file)
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler (logs also shown in terminal)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Change to DEBUG to see all logs

        # Set formatter for both handlers
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
