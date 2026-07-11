import logging


LOGGER_NAME = 'vision.grasp'


def configure_grasp_logger(level=logging.INFO, handlers=None, formatter=None, propagate=False):
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = propagate

    if formatter is None:
        formatter = logging.Formatter('%(message)s')

    if handlers is None:
        handlers = [logging.StreamHandler()]

    if not logger.handlers:
        for handler in handlers:
            if handler.formatter is None:
                handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


def get_grasp_logger():
    return logging.getLogger(LOGGER_NAME)
