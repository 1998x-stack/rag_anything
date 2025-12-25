"""
Centralized logging configuration using loguru.
Provides consistent logging across the entire RAG-Anything framework.
"""

import sys
import traceback
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "500 MB",
    retention: str = "10 days",
    format_string: Optional[str] = None
) -> None:
    """
    Setup loguru logger with consistent formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only console logging
        rotation: When to rotate log file
        retention: How long to keep old log files
        format_string: Custom format string
    """
    # Remove default handler
    logger.remove()
    
    # Default format with colors for console
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # File format without colors
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            backtrace=True,
            diagnose=True,
            enqueue=True  # Thread-safe
        )
    
    logger.info(f"Logger initialized with level={level}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")


def log_exception(exc: Exception, context: str = "") -> None:
    """
    Log exception with full traceback.
    
    Args:
        exc: Exception instance
        context: Additional context string
    """
    tb = traceback.format_exc()
    
    if context:
        logger.error(f"Exception in {context}:")
    
    logger.error(f"{exc.__class__.__name__}: {str(exc)}")
    logger.error(f"Traceback:\n{tb}")


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log function call with parameters.
    
    Args:
        func_name: Name of the function
        **kwargs: Function parameters
    """
    params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Calling {func_name}({params_str})")


def log_performance(operation: str, duration: float, **metrics) -> None:
    """
    Log performance metrics.
    
    Args:
        operation: Name of operation
        duration: Duration in seconds
        **metrics: Additional metrics
    """
    metrics_str = " | ".join(f"{k}={v}" for k, v in metrics.items())
    logger.info(f"Performance [{operation}]: {duration:.3f}s | {metrics_str}")


# Create module-level logger instance
rag_logger = logger.bind(name="RAG-Anything")


# Example usage
if __name__ == "__main__":
    setup_logger(level="DEBUG", log_file=Path("logs/rag_anything.log"))
    
    rag_logger.debug("This is a debug message")
    rag_logger.info("This is an info message")
    rag_logger.warning("This is a warning")
    rag_logger.error("This is an error")
    
    try:
        1 / 0
    except Exception as e:
        log_exception(e, context="main test")