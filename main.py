from src.logger import *
from src.exception import *
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformationClass
from src.components.model_trainer import ModelTrainer


obj = DataIngestion()
train_path,test_path = obj.initiate_data_ingestion()

print(train_path)
print(test_path)

obj2 = DataTransformationClass()
train_arr,test_arr,preprocess_obj_path = obj2.initiate_data_transformation(train_path,test_path)


obj3 = ModelTrainer()
r2_square, best_model_path = obj3.initiate_model_trainer(train_arr,test_arr,preprocess_obj_path)

print(r2_square,best_model_path,sep="\n")