from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
from imblearn.over_sampling import SMOTE # For Resampling imbalanced data

#Moudles related to feature selection
from sklearn.feature_selection import RFE, SelectKBest,f_classif
from sklearn.linear_model import LogisticRegression

## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.utils import save_object

from src.exception import CustomException
from src.logger import logging


## Data Transformation config

@dataclass
class DataTransformationconfig:
    preprocessor_x_file_path=os.path.join('artifacts','preprocessor_num.pkl')
    preprocessor_y_file_path=os.path.join('artifacts','preprocessor_y.pkl')


## Data Ingestionconfig class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation for X variables initiated')

            # Define the column names
            numerical_cols = ['src_bytes','dst_host_serror_rate','count','dst_bytes',
            'same_srv_rate','dst_host_same_srv_rate','logged_in'
            ]


            # Create the preprocessing steps for numerical columns
            numerical_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor_Num = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols)
                ]
            )  

            return  preprocessor_Num

            logging.info('Data Transformation for X variables initiated')

        except Exception as e:
            raise CustomException(e,sys)
        
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_x = self.get_data_transformation_object()

            target_column_name = 'Attack'

            ## features into independent and dependent features
            train_x = train_df.drop(columns=target_column_name,axis=1)
            train_y = train_df[target_column_name]

            test_x = test_df.drop(columns=target_column_name,axis=1)
            test_y = test_df[target_column_name]

            ## Finding Top 10 important features using RFE
            model = LogisticRegression()
            rfe = RFE(estimator=model, n_features_to_select=10)
            rfe = rfe.fit(train_x, train_y)
            imp_vars_RFE = list(train_x.columns[rfe.support_])

            ## Finding Top 10 important features using SKB
            SKB = SelectKBest(f_classif, k=10).fit(train_x,train_y)
            SKB.get_support()
            imp_vars_SKB = list(train_x.columns[SKB.get_support()])

            #Creating a combined list of the top features by intersecting the result of RFC, RFE, SKB
            RFE_features = set(imp_vars_RFE)
            SKB_features = set(imp_vars_SKB)
            Final_Var = RFE_features.intersection(SKB_features)

            ## Selecting the features in train & test dadaset
            X_train_FS = train_x[Final_Var]
            X_test_FS = test_x[Final_Var]

            logging.info("Feature Selection completed")

            logging.info("Intializing resampling of data")

            ## Initializing resampling technique SMOTE under resampler
            resampler = SMOTE(random_state = 42)
            X_train_res, y_train_res = resampler.fit_resample(X_train_FS,train_y)
            logging.info("Resampling of data completed")

            logging.info('Initializing data transformation')

            ## apply the transformation on x & y variables
            x_train_sc = preprocessing_x.fit_transform(X_train_res)
            x_test_sc = preprocessing_x.transform(X_test_FS)

            logging.info('Data transformation completed')

            train_arr = np.c_[x_train_sc, np.array(y_train_res)]
            test_arr = np.c_[x_test_sc, np.array(test_y)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_x_file_path,
                obj=preprocessing_x
            )

            logging.info('Processsor pickle in created and saved')


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_x_file_path
            )            


        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)