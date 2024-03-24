from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

@dataclass(frozen=True)
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Support Vector Machine":SVR(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            model_report = evaluate_models(X_train,y_train,X_test,y_test,models)
            logging.info(f"Model Report: {model_report}")
            #to get best r2_score
            best_model_score = max(list(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            logging.info(f"Best model found on train and test data is {best_model_name}")
            save_object(self.model_trainer_config.trained_model_file_path,best_model)
            logging.info("model saved as pickel")
            best_model.fit(X_train,y_train)
            predicted = best_model.predict(X_test)
            r2_square = r2_score(predicted,y_test)

            return r2_square, self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e,sys)