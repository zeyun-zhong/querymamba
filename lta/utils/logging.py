import logging
from typing import Optional

import lta.utils.distributed as du


def setup_logging(output_dir: Optional[str] = None, level: Optional[str] = 'info',
                  ) -> None:
    """
    Configure logging based on the rank.
    Only the master process logs info. Other processes will only log errors.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
    level = {'info': logging.INFO, 'debug': logging.DEBUG}[level]
    handlers = [logging.StreamHandler()]

    if not du.is_master_proc():
        logging.basicConfig(
            format=_FORMAT,
            datefmt="%m/%d %H:%M:%S",
            level=logging.ERROR,
            handlers=handlers,
        )
        return

    if output_dir is not None:
        handlers.append(logging.FileHandler(f'{output_dir}/logs.log'))

    logging.basicConfig(
        format=_FORMAT,
        datefmt="%m/%d %H:%M:%S",
        level=level,
        handlers=handlers,
    )


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)