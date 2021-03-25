from torch import Tensor

import os
from typing import Callable, Tuple


LOG_FILE_NAME = "log.txt"


def initialise_logging(log_dir: str) -> Tuple[Callable[[str], None], Callable[[Tensor], None]]:
    os.makedirs(log_dir, exist_ok=True)

    def log(message: str):
        """Logs a message by printing it and appending it to a log file."""
        print(message)
        with open(os.path.join(log_dir, LOG_FILE_NAME), 'a+') as log_file:
            log_file.write(message + '\n')

    def log_image(image: Tensor):
        # TODO
        pass

    return log, log_image
