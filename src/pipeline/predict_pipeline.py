import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor_num.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
           
class CustomData:
    def __init__(self,
                 src_bytes:float,
                 dst_host_serror_rate:float,
                 count:float,
                 dst_bytes:float,
                 same_srv_rate:float,
                 dst_host_same_srv_rate:float,
                 logged_in:float):
        
        self.src_bytes=src_bytes
        self.dst_host_serror_rate=dst_host_serror_rate
        self.count=count
        self.dst_bytes=dst_bytes
        self.same_srv_rate=same_srv_rate
        self.dst_host_same_srv_rate=dst_host_same_srv_rate
        self.logged_in = logged_in

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'src_bytes':[self.src_bytes],
                'dst_host_serror_rate':[self.dst_host_serror_rate],
                'count':[self.count],
                'dst_bytes':[self.dst_bytes],
                'same_srv_rate':[self.same_srv_rate],
                'dst_host_same_srv_rate':[self.dst_host_same_srv_rate],
                'logged_in':[self.logged_in]
            }

            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)