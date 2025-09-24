import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import sys

logger = get_logger(__name__)

def read_yaml(file_path:str)->dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file.
    Returns:
        dict: The contents of the YAML file as a dictionary.
    Raises:
        CustomException: If there is an error reading the file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
        logger.info(f"YAML file {file_path} read successfully.")
        return content
    except Exception as e:
        logger.error(f"Error reading YAML file {file_path}: {e}")
        raise CustomException(e, sys)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pandas.DataFrame: The loaded data.
    """
    logger.info(f"Loading CSV file from path: {file_path}")
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        data = pd.read_csv(file_path)
        logger.info(f"CSV file {file_path} loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        raise CustomException("Failded to load data", e)
    

