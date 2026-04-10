#!/usr/bin/env python
# -*- coding:utf-8 _*-
import logging
import sys
import typing
from time_moe.utils.dist_util import is_local_rank_0

# -------- log setting ---------
DEFAULT_LOGGER = "time_moe_logger"

DEFAULT_FORMATTER = logging.Formatter(
    '%(asctime)s - %(filename)s[pid:%(process)d;line:%(lineno)d:%(funcName)s] - %(levelname)s: %(message)s'
)

_ch = logging.StreamHandler(stream=sys.stdout)
_ch.setFormatter(DEFAULT_FORMATTER)

_DEFAULT_HANDLERS = [_ch]

_LOGGER_CACHE = {}  # type: typing.Dict[str, logging.Logger]


def get_logger(name, level="INFO", handlers=None, update=False):
    if name in _LOGGER_CACHE and not update:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = handlers or _DEFAULT_HANDLERS
    logger.propagate = False
    return logger


def log_in_local_rank_0(*msg, type='info', used_logger=None):
    msg = ' '.join([str(s) for s in msg])
    if used_logger is None:
        used_logger = logger

    if is_local_rank_0():
        if type == 'warn' or type == 'warning':
            used_logger.warning(msg)
        elif type == 'error':
            used_logger.error(msg)
        else:
            used_logger.info(msg)


# -------------------------- Singleton Object --------------------------
logger = get_logger(DEFAULT_LOGGER)
