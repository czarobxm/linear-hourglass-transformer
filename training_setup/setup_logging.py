import logging


def setup_logging():
    # Create a logger
    logger = logging.getLogger("basic_logger")
    logger.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")

    # Attach the formatter to the handler
    console_handler.setFormatter(formatter)

    # Attach the handler to the logger
    logger.addHandler(console_handler)

    return logger
