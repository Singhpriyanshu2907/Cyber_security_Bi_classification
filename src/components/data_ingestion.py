import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


## intialize the data ingestion configuration

@dataclass
class DataIngestionconfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
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

            logging.info("Adding attact column in dataset")
            No_Attack['Attack'] = 'No'
            Attack_BufferOverflow['Attack'] = 'Yes'
            Attack_FTPWrite['Attack'] = 'Yes'
            Attack_GuessPassword['Attack'] = 'Yes'
            Attack_Neptune['Attack'] = 'Yes'
            Attack_NMap['Attack'] = 'Yes'
            Attack_PortSweep['Attack'] = 'Yes'
            Attack_RootKit['Attack'] = 'Yes'
            Attack_Satan['Attack'] = 'Yes'
            Attack_Smurf['Attack'] = 'Yes'
            Attack_Normal['Attack'] = 'Yes'
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

            logging.info("Train test split")
            train_set,test_set=train_test_split(sample_data,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )



        except Exception as e:
            logging.info('Error occured in Data Ingestion config')


if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()