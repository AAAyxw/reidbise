# coding: utf-8

import logging.config
import sys, os
import Algorithm.libs.config.model_cfgs as cfgs

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            'format': '%(asctime)s: %(name)s:%(lineno)d - %(levelname)s: %(message)s'
        },
        'standard': {
            'format': '%(asctime)s: %(threadName)s - %(name)s:%(lineno)d - %(levelname)s: %(message)s'
        },
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "CRITICAL",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },

        "file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": os.path.join(cfgs.SAVE_LOG_PATH,'reid_process.log'),
            'when': 'D',
            'interval': 1,
            "backupCount": 15,
            "encoding": "utf8"
        },
    },

    "root": {
        'handlers': ['console', 'file'],
        'level': "INFO",
        'propagate': False
    }
}

if not os.path.exists(cfgs.SAVE_LOG_PATH):
    os.makedirs(cfgs.SAVE_LOG_PATH)
logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name='root'):
    return logging.getLogger(name)

