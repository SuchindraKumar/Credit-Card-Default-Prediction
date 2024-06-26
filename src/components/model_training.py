import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model
from src.utils import save_object

from dataclasses import dataclass
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier



@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

#import logging

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting independent and dependent features from train and test array")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Support Vector": SVC(),
                "KNN Classifier": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost Classifier": XGBClassifier()
            }

            params = {
                "Logistic Regression": {
                    'max_iter': [1000],
                    'solver': ['liblinear']
                },
                "Support Vector": {
                    'kernel': ['rbf'],
                    'C': [10],
                    'degree': [5],
                    'gamma': ['auto'],
                    'class_weight': ['balanced'],
                    'random_state': [42]
                },
                "KNN Classifier": {
                    'n_neighbors': [20]
                },
                "Decision Tree": {
                    'min_samples_split': [15],
                    'min_samples_leaf': [2],
                    'max_features': ['sqrt'],
                    'max_depth': [8],
                    'criterion': ['entropy']
                },
                "Random Forest": {
                    'n_estimators': [200],
                    'max_depth': [10],
                    'min_samples_split': [4],
                    'min_samples_leaf': [5],
                    'max_features': ['sqrt']
                },
                "XGBoost Classifier": {
                    'learning_rate': [0.1],
                    'max_depth': [5],
                    'min_child_weight': [1],
                    'gamma': [0.0],
                    'colsample_bytree': [0.8]
                }
            }

            model_report = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                               models=models, param=params)
            
            # Show scores for all models
            logging.info("Scores for all models:")
            for model_name, metrics in model_report.items():
                logging.info("Model: %s", model_name)
                for metric, value in metrics.items():
                    logging.info("%s: %s", metric, str(value))
                logging.info("\n")

            # Find the best model based on the evaluation report
            best_model_name = max(model_report, key=lambda x: model_report[x]['roc_auc_score'])
            best_model_metrics = model_report[best_model_name]

            # Check if the best model roc_auc_score is below a threshold
            if best_model_metrics['roc_auc_score'] < 0.6:
                raise CustomException("No best model found")

            logging.info("Best found model on both training and testing dataset: %s", best_model_name)
            logging.info("Accuracy of the best model: %f", best_model_metrics['accuracy'])
            logging.info("Evaluation report for the best model:\n%s", best_model_metrics)

            print("\n==========================================\n")
            print("Best found model on both training and testing dataset:", best_model_name)
            print("Accuracy of the best model:", best_model_metrics['accuracy'])
            print("\n==========================================\n")
            # Print evaluation report for the best model
            print("Evaluation report for the best model:")
            for metric, value in best_model_metrics.items():
                print(f"{metric}: {value}")
                
            print("\n***********************************\n")
                

            best_model = models[best_model_name]

            # Save the best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Return the evaluation metrics and the best model
            return model_report, best_model

        except Exception as e:
            logging.info("An exception has occurred in initiate_model_training")
            raise CustomException(e, sys)



