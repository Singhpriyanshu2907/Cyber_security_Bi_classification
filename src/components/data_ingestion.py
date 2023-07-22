import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import get_cols_with_zero_std_dev,treat_outliers

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationconfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


## intialize the data ingestion configuration

@dataclass
class DataIngestionconfig:
    train_set_path=os.path.join('artifacts','train_set.csv')
    test_set_path=os.path.join('artifacts','test_set.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')


## create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            No_Attack = pd.read_csv('Data\Data_of_Attack_Back.csv',index_col=None)
            Attack_BufferOverflow = pd.read_csv('Data\Data_of_Attack_Back_BufferOverflow.csv',index_col=None)
            Attack_FTPWrite = pd.read_csv('Data\Data_of_Attack_Back_FTPWrite.csv',index_col=None)
            Attack_GuessPassword = pd.read_csv('Data\Data_of_Attack_Back_GuessPassword.csv',index_col=None)
            Attack_Neptune = pd.read_csv('Data\Data_of_Attack_Back_Neptune.csv')
            Attack_NMap = pd.read_csv('Data\Data_of_Attack_Back_NMap.csv',index_col=None)
            Attack_Normal = pd.read_csv('Data\Data_of_Attack_Back_Normal.csv',index_col=None)
            Attack_PortSweep = pd.read_csv('Data\Data_of_Attack_Back_PortSweep.csv',index_col=None)
            Attack_RootKit = pd.read_csv('Data\Data_of_Attack_Back_RootKit.csv',index_col=None)
            Attack_Satan = pd.read_csv('Data\Data_of_Attack_Back_Satan.csv',index_col=None)
            Attack_Smurf = pd.read_csv('Data\Data_of_Attack_Back_Smurf.csv',index_col=None)
            logging.info('Dataset read as pandas Dataframe')

            logging.info("Adding Attack column in dataset where 0 represent no attack & 1 otherwise")
            No_Attack['Attack'] = 0
            Attack_BufferOverflow['Attack'] = 1
            Attack_FTPWrite['Attack'] = 1
            Attack_GuessPassword['Attack'] = 1
            Attack_Neptune['Attack'] = 1
            Attack_NMap['Attack'] = 1
            Attack_PortSweep['Attack'] = 1
            Attack_RootKit['Attack'] = 1
            Attack_Satan['Attack'] = 1
            Attack_Smurf['Attack'] = 1
            Attack_Normal['Attack'] = 1
            logging.info("Column conatining target variable added")

            logging.info("Combining datasets into one dataset")
            Dataset = [No_Attack,Attack_BufferOverflow,Attack_FTPWrite,Attack_GuessPassword,Attack_Neptune,Attack_NMap
                                                 ,Attack_PortSweep,Attack_RootKit,Attack_Satan,Attack_Smurf,Attack_Normal]

            Complete_Data = pd.concat(Dataset,axis=0)
            logging.info("Datasets have been combined")


            logging.info("Taking 0.50 of randomly selected data from population data")
            # Set a random seed for reproducibility
            random_seed = 42
            # Perform simple random sampling & selecting sample data for easier computaion
            sample_data = Complete_Data.sample(frac=0.50, random_state=random_seed)
            logging.info("selected 0.50 of data as sample bcz population dataset is too big")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            sample_data.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Intializing Data Cleaning")

            ## Replacing space from coulumn names
            sample_data.rename(columns=lambda x: x.replace(' ', ''), inplace=True)

            ##Dividing the data x variable & target variable
            x = pd.DataFrame(sample_data.drop(columns='Attack'))
            y = pd.DataFrame(sample_data['Attack'])

            # Log the shape of X and Y before any data cleaning or manipulation
            logging.info("Shape of X before data cleaning: {}".format(x.shape))
            logging.info("Shape of Y before data cleaning: {}".format(y.shape))

            ## Dropping the columns having 0 std
            x = pd.DataFrame(get_cols_with_zero_std_dev(x))

            ## ## Treating the outliers
            x = pd.DataFrame(treat_outliers(x))

            # Log the shape of X and Y before any data cleaning or manipulation
            logging.info("Shape of X after data cleaning: {}".format(x.shape))
            logging.info("Shape of Y after data cleaning: {}".format(y.shape))
           
            logging.info("Joining x & y back together")
            sample_data1 = pd.concat([x,y],axis=1)
            logging.info("Shape of sample_data after data cleaning: {}".format(sample_data1.shape))

            logging.info("Data Cleaning completed")

            logging.info("Train test split")
            train_set,test_set=train_test_split(sample_data1,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_set_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_set_path,index=False,header=True)

            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_set_path,
                self.ingestion_config.test_set_path
            )



        except Exception as e:
            raise CustomException(e,sys)
            


if __name__=="__main__":
    obj=DataIngestion()
    train_set,test_set=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_set,test_set)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initate_model_training(train_arr,test_arr))