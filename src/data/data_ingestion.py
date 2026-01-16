import numpy as np
import pandas as pd
import os
import yaml
import logging
from src.logger import logging


#Load params from params.yaml
def load_params(params_path: str) -> dict:

    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.debug("Loaded params from: %s", params_path)
        return params
    except FileNotFoundError:
        logging.error("File not found %s", params_path)
        raise
    except yaml.YAMLError as e:
        logging.error("Yaml error: %s", e)
        raise
    except Exception as e:
        logging.error("Unknown error occured while loading params: %s", e)
        raise


#load data
def load_data(data_uri: str) -> pd.DataFrame:
    '''Load data from the data path'''
    try:
        df = pd.read_csv(data_uri)
        logging.info("data loaded from the uri: %s", data_uri)
        return df
    except pd.errors.ParserError as e:
        logging.error("Error while parsing data: %s",e)
        raise
    except Exception as e:
        logging.error("Unknow error occured while loading data: %s", data_uri)
        raise

def save_data(df: pd.DataFrame, data_path: str) -> None:
    '''Save the data'''
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        df.to_csv(os.path.join(raw_data_path,'data.csv'),index=False)
        logging.info("Data saved in: %s", raw_data_path)
    except Exception as e:
        logging.error("Unknow error occured while saving data: %s", raw_data_path)
        raise

def main():
    try:
        # params = load_params('params.yaml')

        df = load_data(r".\notebooks\retail-data.csv")


        save_data(df,'./data')
    except Exception as e:
        logging.error("Failed to do the data ingestion: %s",e)
        raise

if __name__ == '__main__':
    main()