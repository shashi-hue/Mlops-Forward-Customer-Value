import pandas as pd
import os
import logging
from src.logger import logging
from src.utils import load_params, load_data


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