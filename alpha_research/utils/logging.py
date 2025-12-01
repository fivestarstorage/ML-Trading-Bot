"""
Logging utilities for the Alpha Research Engine.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


_logger: Optional[logging.Logger] = None


def setup_logging(
    name: str = "alpha_research",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for the Alpha Research Engine.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    global _logger
    
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    _logger = logger
    return logger


def get_logger(name: str = "alpha_research") -> logging.Logger:
    """
    Get the configured logger, creating one if necessary.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _logger
    
    if _logger is None:
        _logger = setup_logging(name)
    
    return _logger


class LogContext:
    """Context manager for temporary log level changes."""
    
    def __init__(self, level: int):
        self.level = level
        self.original_level = None
        
    def __enter__(self):
        logger = get_logger()
        self.original_level = logger.level
        logger.setLevel(self.level)
        return logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = get_logger()
        logger.setLevel(self.original_level)
        return False

