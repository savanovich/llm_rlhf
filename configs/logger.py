import logging
import sys


def setup_log(filename='logs/preprocessing.log', level=logging.INFO):
    file_handler = logging.FileHandler(filename=filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

