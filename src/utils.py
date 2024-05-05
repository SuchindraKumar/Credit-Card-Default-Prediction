import os
import sys
import pickle

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Exception occurred in save_object method")
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            # Get hyperparameters for the current model
            model_params = param[model_name]

            # Perform grid search cross-validation to find the best hyperparameters
            gs = GridSearchCV(model, model_params, cv=3)
            gs.fit(X_train, y_train)

            # Update model with best parameters and fit to training data
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions on training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_test_pred)
            confusion = confusion_matrix(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            roc_auc = roc_auc_score(y_test, y_test_pred)

            # Store evaluation metrics in the report dictionary
            report[model_name] = {
                'accuracy': accuracy,
                'confusion_matrix': confusion,
                'f1_score': f1,
                'precision_score': precision,
                'recall_score': recall,
                'roc_auc_score': roc_auc
            }

        return report

    except Exception as e:
        logging.info("Exception occurred in evaluate_model function")
        raise CustomException(e, sys)
    



def load_object(file_path):
    try:

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Exception occurred in loading object")
        raise CustomException(e, sys)