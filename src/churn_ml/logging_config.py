import logging
import os
import sys
from typing import Optional

try:
    from pythonjsonlogger import json as jsonlogger
except ImportError:
    try:
        from pythonjsonlogger import jsonlogger
    except ImportError:
        jsonlogger = None


def setup_logger(
    name: str = "churn_ml",
    log_level: Optional[str] = None,
    as_json: Optional[bool] = None,
) -> logging.Logger:
    """
    Configures and returns a structured logger.
    Supports standard stream logging or JSON formatted logging for cloud/production environments.
    """
    level = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    json_logging = as_json if as_json is not None else (os.getenv("LOG_FORMAT", "text").lower() == "json")

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level, logging.INFO))

        if json_logging and jsonlogger is not None:
            formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        else:
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(module)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = setup_logger()
