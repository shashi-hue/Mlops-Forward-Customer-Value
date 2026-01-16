import logging
import yaml
import pandas as pd
import numpy as np
import pickle
import json

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

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise


from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
# Helper functions
def evaluate_regression(y_true, y_pred):
    return {
        "rmse_log": root_mean_squared_error(y_true, y_pred),
        "mae_log": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }


def inverse_rmse(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return root_mean_squared_error(y_true, y_pred)

from scipy.stats import spearmanr

def spearman_rank(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise