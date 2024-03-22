from src.logger import *
from src.exception import *
import sys

try:
    a = 10/0
except Exception as e:
    logging.info(f"Divide By Zero")
    raise CustomException(e,sys)

