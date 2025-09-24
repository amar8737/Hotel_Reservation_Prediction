# src/data_preprocessing.py
import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import sys
import numpy as np
from utils.common_functions import read_yaml, load_data
from config.paths_config import TEST_FILE_PATH, TRAIN_FILE_PATH, PROCESSED_DIR, PROCESSED_TEST_FILE_PATH,\
      PROCESSED_TRAIN_FILE_PATH, CONFIG_FILE_PATH
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)


class DataPreprocessor:

    def __init__(self,train_path:str,test_path:str,config_path:str,processed_dir:str):
        """
        Initializes the DataPreprocessor with paths and configuration.
        
        """
        self.processed_dir = processed_dir
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)
        self.train_path = train_path
        self.test_path = test_path
        self.config_path = config_path
        self.config = read_yaml(self.config_path)
        # MODIFICATION: Added attributes to fix saving error
        self.processed_train_path = PROCESSED_TRAIN_FILE_PATH
        self.processed_test_path = PROCESSED_TEST_FILE_PATH

    def preprocess_data(self, df):
        """
        Preprocess the training and testing data.
        """
        try:
            logger.info("Starting data preprocessing...")
            
            # --- Initial Cleaning ---
            df.drop(columns=['Booking_ID'], inplace=True, errors='ignore')
            df.drop_duplicates(keep='first', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            cat_cols = self.config['data_processing']['categorical_features']
            num_cols = self.config['data_processing']['numerical_features']

            # --- STEP 1: Impute Missing Values FIRST ---
            
            # Impute CATEGORICAL columns with the most frequent value (mode)
            logger.info("Handling missing values in categorical columns...")
            for col in cat_cols:
                if col in df.columns and df[col].isnull().any():
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)

            # Impute NUMERICAL columns with the median
            logger.info("Handling missing values in numerical columns...")
            for col in num_cols:
                if col in df.columns and df[col].isnull().any():
                    median_val = df[col].median()
                    # ✅ FIX: Use safer reassignment instead of inplace=True
                    df[col] = df[col].fillna(median_val)
            
            logger.info("All missing values handled. ✅")

            # --- STEP 2: Encode Categorical Columns ---
            logger.info(f"Encoding categorical columns: {cat_cols}")
            self.label_encoder = LabelEncoder()
            for col in cat_cols:
                if col in df.columns:
                    df[col] = self.label_encoder.fit_transform(df[col])
            
            # ✅ FIX: The problematic .map() loop has been removed.

            # --- STEP 3: Skewness Correction ---
            logger.info("Doing skewness correction using log transformation...")
            skew_threshold = self.config['data_processing']['skewness_threshold']
            skewness = df[num_cols].skew()
            skewed_cols = skewness[abs(skewness) > skew_threshold].index
            
            if not skewed_cols.empty:
                logger.info(f"Skewed columns: {list(skewed_cols)}")
                df[skewed_cols] = np.log1p(df[skewed_cols])
                logger.info("Skewness correction completed.")
            else:
                logger.info("No skewed columns found to correct.")

            logger.info("Data preprocessing completed.")
            return df
            
        except Exception as e:
            logger.error(f"Error occurred during data preprocessing: {e}")
            raise CustomException("Error occurred during data preprocessing", e)
        
    def balance_data(self,df,target_column:str):
        try:
            logger.info("Starting data balancing...")
            self.target_column = target_column
            self.smote = SMOTE(random_state=42)
            
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            print("Missing values in X before SMOTE:")
            print(X.isnull().sum())
            logger.info(f"Original class distribution:\n{y.value_counts()}")
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
            df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
            df_balanced[self.target_column] = y_resampled
            logger.info(f"Balanced class distribution:\n{df_balanced[self.target_column].value_counts()}")
            logger.info("Data balancing completed.")
            return df_balanced
        except Exception as e:
            logger.error(f"Error occurred while balancing data: {e}")
            raise CustomException("Error occurred while balancing data", e)

    def select_features(self,df,target_column:str):
        try:
            logger.info("Starting feature selection...")
            self.target_column = target_column
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            self.model = RandomForestClassifier(random_state=42)
            self.model.fit(X, y)
            feature_importances = pd.Series(self.model.feature_importances_, index=X.columns)
            feature_importances = feature_importances.sort_values(ascending=False)
            logger.info(f"Feature importances:\n{feature_importances}")
            num_features_to_select = self.config['data_processing']['no_of_features'] # Corrected the key
            feature_importances = feature_importances.head(num_features_to_select)
            selected_features = feature_importances.index.tolist()
            logger.info(f"Selected features: {selected_features}")
            df_selected = df[selected_features + [self.target_column]]
            logger.info("Feature selection completed successfully.")
            return df_selected
        except Exception as e:
            logger.error(f"Error occurred while selecting features: {e}")
            raise CustomException("Error occurred while selecting features", e)
    
    def process(self):
        try:
            logger.info("Loading training and testing data...")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)
            logger.info("Preprocessing training data...")
            train_df = self.preprocess_data(train_df)
            logger.info("Preprocessing testing data...")
            test_df = self.preprocess_data(test_df)
            target_column = self.config['data_processing']['target_column']

                # ---> ADD THIS DEBUGGING BLOCK HERE <---
            logger.info("Checking for NaNs right before balancing...")
            nan_counts = train_df.isnull().sum()
            columns_with_nans = nan_counts[nan_counts > 0]
            if not columns_with_nans.empty:
                logger.warning(f"Columns still containing NaNs:\n{columns_with_nans}")
            else:
                logger.info("No NaNs found before balancing. The issue might be elsewhare.")
            # ------------------------------------

            logger.info("Balancing training data...")
            train_df = self.balance_data(train_df, target_column)
            
            # --- MODIFICATION: Critical data leakage fix. DO NOT balance the test set. ---
            # logger.info("balancing testing data...")
            # test_df = self.balance_data(test_df, target_column)
            
            logger.info("Selecting features from training data...")
            train_df = self.select_features(train_df, target_column)
            logger.info("Applying selected features to testing data...") # Modified log
            test_df = test_df[train_df.columns]
            train_df.to_csv(self.processed_train_path, index=False)
            test_df.to_csv(self.processed_test_path, index=False)
            logger.info(f"Processed training data saved to {self.processed_train_path}")
            logger.info(f"Processed testing data saved to {self.processed_test_path}")
            logger.info("Data processing pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Error occurred during the processing pipeline: {e}")
            raise CustomException("Error occurred during the processing pipeline", e)
            
if __name__ == "__main__":
    print(TRAIN_FILE_PATH, TEST_FILE_PATH, CONFIG_FILE_PATH, PROCESSED_DIR)
    try:
        data_preprocessor = DataPreprocessor(
            train_path=TRAIN_FILE_PATH,
            test_path=TEST_FILE_PATH,
            config_path=CONFIG_FILE_PATH,
            processed_dir=PROCESSED_DIR
        )
        data_preprocessor.process()
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise CustomException("Error in main execution", e)