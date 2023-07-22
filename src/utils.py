import os
import sys
import pickle
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import accuracy_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    

## Below function will be used to drop Variables with std_dev 0 as those variable wont contribute much in predictions
def get_cols_with_zero_std_dev(df: pd.DataFrame):
    try:
        cols_to_drop = []
        num_cols = [col for col in df.columns if df[col].dtype != 'O']  # numerical cols only
        for col in num_cols:
            if df[col].std() == 0:
                cols_to_drop.append(col)
        return df.drop(columns=cols_to_drop)
    except Exception as e:
        logging.info('Exception Occured in get_cols_with_zero_std_dev function utils')
        raise CustomException(e,sys)
    

def treat_outliers(df, lower_quantile=0.01, upper_quantile=0.99):
    try:
        treated_df = pd.DataFrame()

        for column in df.columns:
            lower_threshold = df[column].quantile(lower_quantile)
            upper_threshold = df[column].quantile(upper_quantile)

            treated_column = df[column].clip(lower_threshold, upper_threshold)
            treated_df[column] = treated_column

        return treated_df
    except Exception as e:
        logging.info('Exception Occured in treat_outliers function utils')
        raise CustomException(e,sys)





def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        results = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(x_train, y_train)
            
            # Make predictions on test and train data
            y_pred_test = model.predict(x_test)
            y_pred_train = model.predict(x_train)
            
            # Calculate accuracy for test and train data
            accuracy_test = accuracy_score(y_test, y_pred_test)
            accuracy_train = accuracy_score(y_train, y_pred_train)
            
            results[list(models.keys())[i]] =  accuracy_test       
        return results
    
    except Exception as e:
        logging.info('Exception Occured in evaluate models function utils')
        raise CustomException(e,sys)

