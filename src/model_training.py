import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from src.logger import get_logger
from sklearn.preprocessing import LabelEncoder
from src.custom_exception import CustomException
import sys
from config.paths_config import MODEL_DIR, PROCESSED_TRAIN_FILE_PATH, PROCESSED_TEST_FILE_PATH,CONFIG_FILE_PATH
from config.model_params import LIGHTGBM_PARAMS, CATBOOST_PARAMS, RF_PARAMS, RANDOM_SEARCH_PARAMS
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score,f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from utils.common_functions import load_data, read_yaml
from scipy.stats import uniform, randint
import mlflow
import mlflow.sklearn


logger = get_logger(__name__)

class ModelTrainer:

    def __init__(self, processed_train_path: str, processed_test_path: str, model_output_path: str):
        """
        Initializes the ModelTrainer with paths to processed data and model output.
        Args:
            processed_train_path (str): Path to the processed training data CSV file.
            processed_test_path (str): Path to the processed testing data CSV file.
            model_output_path (str): Path to save the trained model.
        """
        self.processed_train_path = processed_train_path
        self.processed_test_path = processed_test_path
        self.model_output_path = model_output_path
        self.model_dir = os.path.dirname(model_output_path)
        self.label_encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        os.makedirs(self.model_dir, exist_ok=True)
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self, target_column: str):
        """
        Loads the processed data and splits it into features and target.
        Returns:
            X_train, y_train: Split features and target datasets.
        """
        try:
            logger.info("Loading processed data...")
            train_df = load_data(self.processed_train_path)
            test_df = load_data(self.processed_test_path)
            X_train = train_df.drop(columns=[target_column], errors='ignore')
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column], errors='ignore')
            y_test = test_df[target_column]
            logger.info("Encoding target labels...")
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
            logger.info("Data loaded and split into features and target.")
            return X_train, y_train, X_test, y_test, label_encoder
        except Exception as e:
            logger.error(f"Error loading and splitting data: {e}")
            raise CustomException("Failed to load and split data", e)
    
    def train_lgbm(self, X_train, y_train):
        """
        Trains a LightGBM model with hyperparameter tuning using RandomizedSearchCV.
        Returns:
            best_model: The best trained LightGBM model.
        """
        try:
            logger.info("Training LightGBM model with hyperparameter tuning...")
            lgbm = LGBMClassifier(random_state=self.random_search_params.get('random_state', 42))
            random_search = RandomizedSearchCV(
                estimator=lgbm,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                scoring=self.random_search_params['scoring'],
                cv=self.random_search_params['cv'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                n_jobs=self.random_search_params['n_jobs']
            )
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            logger.info(f"Best parameters found: {random_search.best_params_}")
            return best_model
        except Exception as e:
            logger.error(f"Error training LightGBM model: {e}")
            raise CustomException("Failed to train LightGBM model", e)
    def evaluate_model(self, model, X_test, y_test,label_encoder=None):
        """
        Evaluates the trained model on the test set and prints classification report.
        Args:
            model: The trained model to evaluate.
            X_test: Test features.
            y_test: Test target.
        """
        try:
            logger.info("Evaluating model...")
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
            logger.info(f"Classification Report:\n{report}")
            
            # <<< FIX: Use the sklearn function for consistency
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0.0)
            rec = recall_score(y_test, y_pred, zero_division=0.0)
            f1 = f1_score(y_test, y_pred, zero_division=0.0)
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            logger.info(f"Accuracy: {acc}")
            logger.info(f"Precision: {prec}")
            logger.info(f"Recall: {rec}")
            logger.info(f"F1 Score: {f1}")
            logger.info(f"ROC AUC Score: {roc_auc}")
            return {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc_auc
            }
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e)

    def save_artifact(self, model,label_encoder):
        """
        Saves the trained model and label encoder to the specified output path.
        Args:
            model: The trained model to save.
            label_encoder: The label encoder to save.
        """
        try:
            logger.info(f"Saving model to {self.model_output_path}...")
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)
            joblib.dump(label_encoder, self.label_encoder_path)
            logger.info(f"Model saved successfully to {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise CustomException("Failed to save model", e)
        
    def run(self):
        """
        Runs the model training and evaluation pipeline.
        Args:
            target_column: The target column for prediction.
        """
        try:
            with mlflow.start_run():
                logger.info("Starting model training...")
                params = read_yaml(CONFIG_FILE_PATH)
                target_column = params['data_processing']['target_column']
                X_train, y_train, X_test, y_test, label_encoder = self.load_and_split_data(target_column)
                model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(model, X_test, y_test, label_encoder)
                self.save_artifact(model, label_encoder)
                mlflow.sklearn.log_model(model, "model")
                mlflow.log_param("model_type", "LightGBM")
                mlflow.log_artifacts(CONFIG_FILE_PATH, artifact_path="config")  
                mlflow.log_artifacts(self.processed_train_path, artifact_path="data")  
                mlflow.log_artifacts(self.processed_test_path, artifact_path="data")
                mlflow.log_artifact(self.model_output_path)
                mlflow.log_artifact(self.label_encoder_path)
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                logger.info("Model training pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException("Failed to run model training pipeline", e)
        

if __name__ == "__main__":
    trainer = ModelTrainer(
        processed_train_path=PROCESSED_TRAIN_FILE_PATH,
        processed_test_path=PROCESSED_TEST_FILE_PATH,
        model_output_path=os.path.join(MODEL_DIR, 'lgbm_model.pkl')
    )
    trainer.run()