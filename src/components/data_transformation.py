import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object

#dataingestion is the process of collecting and preparing data for analysis or modeling.
#it involves gathering data from various sources, cleaning and transforming it into a suitable format for analysis.
@dataclass
class DataTransformationConfig:
    prepossor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        

    def get_data_transformer_object(self):
        # This method is responsible for data transformation, including encoding categorical variables and scaling numerical features.
        # It returns a preprocessor object that can be used to transform the data.
        try:
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            
            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False, with_std=True))
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputor', SimpleImputer(strategy='most_frequent')),
                ('OneHotEncoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False, with_std=True))
            ])
            logging.info("Numerical and categorical columns encoding completed")

            # Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            logging.info("Preprocessor object created")
            return preprocessor
                
    
        except Exception as e:
            raise CustomException(e, sys) 
        
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data transformation initiated")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            
            logging.info("Train and test dataframes read successfully")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = 'math score'
            numerical_columns = ['writing score', 'reading score']
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing dataframes")
            # Transform the input features using the preprocessor object

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Preprocessing completed")

            save_object(
                file_path=self.data_transformation_config.prepossor_obj_file_path, 
                obj=preprocessor_obj
            )
            # Save the preprocessor object to a file    

    
        except Exception as e:
            raise CustomException(e, sys)               