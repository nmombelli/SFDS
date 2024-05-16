import logging


class CustomFormatter(logging.Formatter):

    grey = "\x1b[37;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def set_logger(level: str = 'DEBUG'):
    """
    Configure the logger for the project
    :param level: allowed minimum levels are DEBUG and INFO.
    :return:
    """

    if level == 'DEBUG':
        logging.root.setLevel(logging.DEBUG)
    elif level == 'INFO':
        logging.root.setLevel(logging.INFO)
    elif level == 'WARNING':
        logging.root.setLevel(logging.WARNING)
    elif level == 'ERROR':
        logging.root.setLevel(logging.ERROR)
    else:
        raise ValueError('Log level not allowed')

    # create console handler with a higher log level
    ch = logging.StreamHandler()

    if level == 'DEBUG':
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)

    ch.setFormatter(CustomFormatter())

    logging.root.addHandler(ch)

    return
