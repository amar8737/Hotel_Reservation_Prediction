if __name__ == "__main__":
    from model_training import ModelTrainer
    from config.paths_config import PROCESSED_TRAIN_FILE_PATH, PROCESSED_TEST_FILE_PATH, MODEL_DIR
    trainer = ModelTrainer(
        processed_train_path=PROCESSED_TRAIN_FILE_PATH,
        processed_test_path=PROCESSED_TEST_FILE_PATH,
        model_output_path=os.path.join(MODEL_DIR, 'lgbm_model.pkl')
    )
    trainer.run()