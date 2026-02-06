"""
Logging Configuration and Utilities
Centralized logging with structured output and context
"""
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
import json


class LoggerConfig:
    """Logger configuration and setup"""
    
    _initialized = False
    
    @classmethod
    def setup(
        cls,
        log_level: str = "INFO",
        log_file: Optional[Path] = None,
        json_logs: bool = False,
        enable_console: bool = True,
    ) -> None:
        """
        Configure application-wide logging
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for logging
            json_logs: Enable JSON-formatted logs
            enable_console: Enable console output
        """
        if cls._initialized:
            return
        
        # Remove default handler
        logger.remove()
        
        # Console handler
        if enable_console:
            if json_logs:
                logger.add(
                    sys.stderr,
                    format=cls._json_formatter,
                    level=log_level,
                    serialize=True,
                )
            else:
                logger.add(
                    sys.stderr,
                    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                           "<level>{level: <8}</level> | "
                           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                           "<level>{message}</level>",
                    level=log_level,
                    colorize=True,
                )
        
        # File handler
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                str(log_file),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                level=log_level,
                rotation="100 MB",
                retention="30 days",
                compression="zip",
            )
        
        cls._initialized = True
    
    @staticmethod
    def _json_formatter(record):
        """Format log record as JSON"""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
        }
        
        if record["exception"]:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback,
            }
        
        return json.dumps(log_entry)


def get_logger(name: str):
    """
    Get a logger instance for a module
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logger.bind(module=name)


# Initialize with defaults
LoggerConfig.setup()
