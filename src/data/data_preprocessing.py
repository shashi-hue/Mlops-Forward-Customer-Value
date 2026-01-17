import numpy as np
import pandas as pd
import os
from src.logger import logging

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    '''data preprocessing'''
    try:
        logging.info("Preprocessing started")
        df = df.dropna(subset=['Customer ID'])

        df = df[~df['Invoice'].str.startswith('C')]

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


        data_path = os.path.join('./data','interim')
        os.makedirs(data_path, exist_ok=True)

        df.to_csv(os.path.join(data_path,'data.csv'),index=False)

        logging.info("Processed data saved into: %s",data_path)
    except Exception as e:
        logging.error("Error occured in data_preprocessing: %s",e)
        raise

if __name__ == '__main__':
    main()