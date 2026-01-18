import json
import logging
import os
import mlflow
import mlflow.sklearn
import dagshub
from src.logger import logging
from src.utils import load_model, load_data, evaluate_regression, inverse_rmse, spearman_rank


#For local use 
# dagshub.init(repo_owner='shashi-hue', repo_name='Mlops-Forward-Customer-Value', mlflow=True)

#For production use
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST env variable not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "shashi-hue"
repo_name = "Mlops-Forward-Customer-Value"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')



def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

# def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
#     """Save the model run ID and path to a JSON file."""
#     try:
#         model_info = {'run_id': run_id, 'model_path': model_path}
#         with open(file_path, 'w') as file:
#             json.dump(model_info, file, indent=4)
#         logging.debug('Model info saved to %s', file_path)
#     except Exception as e:
#         logging.error('Error occurred while saving the model info: %s', e)
#         raise

def main():
    mlflow.set_experiment("pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            rf_model = load_model('./models/rf_model.pkl')
            test_data = load_data('./data/processed/test_data.csv')
            X_test = test_data.drop(columns=['target_clv'])
            y_test = test_data['target_clv']

            y_pred = rf_model.predict(X_test)

            metrics = evaluate_regression(y_test, y_pred)
            metrics["rmse_currency"] = inverse_rmse(y_test, y_pred)
            metrics["spearman_rank"] = spearman_rank(y_test, y_pred)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            save_metrics(metrics, 'reports/metrics.json')

            if hasattr(rf_model, 'get_params'):
                params = rf_model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            mlflow.sklearn.log_model(
                rf_model,
                artifact_path="model",
                registered_model_name="my_model"
            )


            # save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

            mlflow.log_artifact('reports/metrics.json')
        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()