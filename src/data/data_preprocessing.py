import numpy as np
import pandas as pd
import os
import yaml
from src.logger import logging
from sklearn.model_selection import train_test_split

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

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    '''data preprocessing'''
    try:
        logging.info("Preprocessing started")
        df = df.dropna(subset=['Customer ID'])

        df = df[~df['Invoice'].str.startswith('C')]

        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) 

        df['Total Amount'] = df['Price'] * df['Quantity']

        return df
    except Exception as e:
        logging.error("Error while doing preprocessing: %s",e)
        raise

def main():
    try:
        df = pd.read_csv('./data/raw/data.csv')
        logging.info("Data loaded properly")
        df = preprocessing(df)
        logging.info("preprocessing completed")

        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']

        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

        data_path = os.path.join('./data','interim')
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(data_path,'train_data.csv'),index=False)
        test_data.to_csv(os.path.join(data_path,'test_data.csv'),index=False)

        logging.info("Processed train and test data saved into: %s",data_path)
    except Exception as e:
        logging.error("Error occured in data_preprocessing: %s",e)
        raise

if __name__ == '__main__':
    main()