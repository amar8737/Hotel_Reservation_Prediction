from src.data_ingestion import DataIngestion
from config.paths_config import RAW_FILE_PATH, RAW_DIR, PROCESSED_DIR, PROCESSED_TRAIN_FILE_PATH, \
    PROCESSED_TEST_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH, CONFIG_FILE_PATH,MODEL_DIR
from utils.common_functions import read_yaml
from src.model_training import ModelTrainer
import os


if __name__ == "__main__":

    ### 1. Data Ingestion
    config_content = read_yaml(CONFIG_FILE_PATH)
    data_ingestion_pipeline = DataIngestion(config=config_content)
    train_data_path, test_data_path = data_ingestion_pipeline.run()

    ### 2. Model Training
    model_trainer = ModelTrainer(
        processed_train_path=PROCESSED_TRAIN_FILE_PATH,
        processed_test_path=PROCESSED_TEST_FILE_PATH,
        model_output_path=os.path.join(MODEL_DIR, 'lgbm_model.pkl')
    )
    model_trainer.run()