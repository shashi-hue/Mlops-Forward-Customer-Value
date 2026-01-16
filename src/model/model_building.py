import pandas as pd
from src.logger import logging
from sklearn.ensemble import RandomForestRegressor
from src.utils import load_params, load_data
import pickle

params = load_params('params.yaml')
n_estimators = params['random_forest']['n_estimators']
max_depth = params['random_forest']['max_depth']
min_samples_leaf = params['random_forest']['min_samples_leaf']
max_features = params['random_forest']['max_features']
random_state = params['random_forest']['random_state']
min_samples_split = params['random_forest']['min_samples_split']

def model_traing(X_train: pd.DataFrame, y_train: pd.DataFrame) -> RandomForestRegressor:
    '''Train Random Forest model'''
    try:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        rf.fit(X_train,y_train)
        logging.info("Model training completed")
        return rf
    except Exception as e:
        logging.error("Error while traing the model: %s", e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        train_data = load_data('./data/processed/train_data.csv')
        X_train = train_data.drop(columns=['target_clv'])
        y_train = train_data['target_clv']

        rf = model_traing(X_train, y_train)

        save_model(rf, 'models/rf_model.pkl')
    except Exception as e:
        logging.error("Error building model: %s", e)
        raise

if __name__ == '__main__':
    main()