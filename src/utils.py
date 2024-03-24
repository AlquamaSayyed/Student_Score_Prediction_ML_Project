from src.logger import logging
from src.exception import CustomException
import os
import sys
import pickle
import dill
from sklearn.metrics import r2_score
from ensure import ensure_annotations
def save_object(filepath,obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e,sys)

@ensure_annotations
def evaluate_models(x_train,y_train,x_test,y_test,models:dict) ->dict:
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train_pred,y_train)
            test_model_score = r2_score(y_test_pred, y_test)

            report[(list(models.keys())[i])] = test_model_score

        return  report

    except Exception as e:
        raise CustomException(e,sys)

def load_object(filepath):
    try:
        dir_path = os.path.dirname(filepath)
        with open(filepath, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)
