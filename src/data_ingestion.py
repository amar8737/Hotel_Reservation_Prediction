# src/data_ingestion.py

import os
import sys

import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split

# REFACTORED: Combined and cleaned up imports to remove duplicates.
from config.paths_config import (CONFIG_FILE_PATH, RAW_DIR, RAW_FILE_PATH,
                                 TEST_FILE_PATH, TRAIN_FILE_PATH)
from src.custom_exception import CustomException
from src.logger import get_logger
from utils.common_functions import read_yaml

# Initialize the logger at the module level for consistency.
logger = get_logger(__name__)

class DataIngestion:
    """
    Handles downloading data from Google Cloud Storage and splitting it into
    training and testing datasets.
    """
    def __init__(self, config: dict):
        """
        Initializes the DataIngestion component.

        Args:
            config (dict): A dictionary containing configuration parameters.
        """
        try:
            self.config = config['data_ingestion']
            self.bucket_name = self.config['bucket_name']
            self.source_blob_name = self.config['source_blob_name']
            self.train_ratio = self.config['train_ratio']

            # Create the necessary directory for raw data artifacts.
            os.makedirs(RAW_DIR, exist_ok=True)
            logger.info(f"Artifacts directory '{RAW_DIR}' is ready.")
            logger.info("DataIngestion component initialized successfully.")

        except KeyError as e:
            logger.error(f"Missing required key in configuration: {e}")
            raise CustomException(f"Configuration error: Missing key {e}", sys)
        except Exception as e:
            logger.error(f"Error during DataIngestion initialization: {e}")
            raise CustomException(e, sys)

    def download_data_from_gcs(self) -> None:
        """
        Downloads the source data file from the specified GCS bucket.
        """
        try:
            logger.info(f"Downloading data from GCS bucket '{self.bucket_name}'...")
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.source_blob_name)
            
            blob.download_to_filename(RAW_FILE_PATH)
            
            logger.info(f"Successfully downloaded '{self.source_blob_name}' to '{RAW_FILE_PATH}'")
        except Exception as e:
            # REFACTORED: Improved error logging to always include the actual exception.
            logger.error(f"Failed to download data from GCS. Error: {e}")
            raise CustomException(e, sys)

    def split_data_into_train_test(self) -> None:
        """
        Reads the raw data, splits it into training/testing sets, and saves them.
        """
        try:
            logger.info("Reading raw data and splitting into train/test sets...")
            df = pd.read_csv(RAW_FILE_PATH)
            
            # test_size = 1.0 - train_ratio ensures the split is correct.
            train_set, test_set = train_test_split(
                df, 
                test_size=(1 - self.train_ratio), 
                random_state=42
            )

            train_set.to_csv(TRAIN_FILE_PATH, index=False, header=True)
            test_set.to_csv(TEST_FILE_PATH, index=False, header=True)

            logger.info(f"Data split complete. Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")
            logger.info(f"Train data saved to: {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to: {TEST_FILE_PATH}")
        except FileNotFoundError:
            logger.error(f"Raw data file not found at '{RAW_FILE_PATH}'. Cannot split data.")
            raise CustomException(f"Raw data file not found at '{RAW_FILE_PATH}'", sys)
        except Exception as e:
            logger.error(f"Failed to split data. Error: {e}")
            raise CustomException(e, sys)

    def run(self) -> tuple[str, str]:
        """
        Executes the full data ingestion pipeline: download and split.

        Returns:
            tuple[str, str]: File paths for the training and testing data.
        """
        logger.info(">>>>> Starting Data Ingestion pipeline <<<<<")
        try:
            self.download_data_from_gcs()
            self.split_data_into_train_test()
            return (TRAIN_FILE_PATH, TEST_FILE_PATH)
        except Exception as e:
            # The specific exception is already logged by the failed method.
            # This message provides context for the overall pipeline failure.
            logger.error("Data Ingestion pipeline failed.")
            raise CustomException("Data Ingestion pipeline failed", sys) from e
        finally:
            logger.info(">>>>> Data Ingestion pipeline finished <<<<<")


if __name__ == "__main__":
    try:
        config_content = read_yaml(CONFIG_FILE_PATH)
        data_ingestion_pipeline = DataIngestion(config=config_content)
        train_data_path, test_data_path = data_ingestion_pipeline.run()
        logger.info(f"Pipeline completed successfully. Train data at: {train_data_path}, Test data at: {test_data_path}")
    except Exception as e:
        # This is a final catch-all for any unhandled exceptions.
        logger.critical("The main execution of the data ingestion script failed catastrophically.")