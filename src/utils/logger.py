"""Logging setup for the data pipeline."""
import logging
import sys


def setup_logger(name: str = "argus_quant", level: str | None = None) -> logging.Logger:
    """Configures and returns a consistent logger."""
    from src.config.settings import get_settings

    if level is None:
        level = get_settings().log_level

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
