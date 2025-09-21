# sorted start
from config.paths_config import RAW_DIR, RAW_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH
from config.paths_config import CONFIG_FILE_PATH, RAW_DIR, TRAIN_FILE_PATH, TEST_FILE_PATH
from google.cloud import storage
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException 
import sys
from utils.common_functions import read_yaml
# sorted end

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        try:
            self.config = config['data_ingestion']
            self.bucket_name = self.config['bucket_name']
            self.train_test_ratio = self.config['train_ratio']
            self.source_blob_name = self.config['source_blob_name']
            os.makedirs(RAW_DIR, exist_ok=True)
            logger.info(f"DataIngestion directory {RAW_DIR} is ready.")
            logger.info(f"DataIngestion initialized with bucket: {self.bucket_name}, source_blob: {self.source_blob_name}, train_ratio: {self.train_test_ratio}")
        except Exception as e:
            logger.error(f"Error initializing DataIngestion: {e}")
            raise CustomException(e, sys)

    def download_data(self):
        """Downloads data from a Google Cloud Storage bucket."""
        try:
            if not os.path.exists(RAW_DIR):
                os.makedirs(RAW_DIR)
                logger.info(f"Created directory: {RAW_DIR}")

            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.source_blob_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"Downloaded {self.source_blob_name} from bucket {self.bucket_name} to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise CustomException(e, sys)

    def split_data(self):
        """Splits the raw data into training and testing sets."""
        try:
            df = pd.read_csv(RAW_FILE_PATH)
            train_set, test_set = train_test_split(df, train_size=self.train_ratio, random_state=42)

            train_set.to_csv(TRAIN_FILE_PATH, index=False)
            test_set.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Data split into train and test sets with ratio {self.train_ratio}.")
            logger.info(f"Training set saved to {TRAIN_FILE_PATH}")
            logger.info(f"Testing set saved to {TEST_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise CustomException(e, sys)
    def download_csv_from_gcp(self):
        """Downloads CSV file from GCP bucket."""
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.source_blob_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"Downloaded {self.source_blob_name} from bucket {self.bucket_name} to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error downloading CSV from GCP")
            raise CustomException("Error downloading CSV from GCP", e)

    def split_data(self):
        """Splits the raw data into training and testing sets."""
        try:
            logger.info("Starting data split process.")
            df = pd.read_csv(RAW_FILE_PATH)
            train_set, test_set = train_test_split(df, test_size=1-self.train_test_ratio, random_state=42)

            train_set.to_csv(TRAIN_FILE_PATH, index=False)
            test_set.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Data split into train and test sets with ratio {self.train_test_ratio}.")
            logger.info(f"Training set saved to {TRAIN_FILE_PATH}")
            logger.info(f"Testing set saved to {TEST_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error splitting data")
            raise CustomException(e, sys) 

    def run(self):
        """Initiates the data ingestion process."""
        try:
            logger.info("Starting data ingestion process.")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion process completed successfully.")
            return TRAIN_FILE_PATH, TEST_FILE_PATH
        except Exception as e:
            logger.error(f"Error in data ingestion process")
            raise CustomException(e, sys)
        finally:
            logger.info("Data ingestion process finished.")


if __name__ == "__main__":
    try:
        config = read_yaml(CONFIG_FILE_PATH)
        data_ingestion = DataIngestion(config)
        train_path, test_path = data_ingestion.run()
        logger.info(f"Data ingestion completed. Train path: {train_path}, Test path: {test_path}")
    except Exception as e:
        logger.error(f"Error in main execution")
        raise CustomException(e, sys)
