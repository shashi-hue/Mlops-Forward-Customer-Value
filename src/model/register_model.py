import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub
from src.utils import load_model_info

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# os.environ["MLFLOW_TRACKING_USERNAME"] = 'shashi-hue'
# os.environ["MLFLOW_TRACKING_PASSWORD"] = '78e6431c64e560d8dfa4952bd9c0d716c59d0825'
dagshub.init(repo_owner='shashi-hue', repo_name='Mlops-Forward-Customer-Value', mlflow=True)


def register_model(model_name: str, model_info: dict, alias: str = "candidate"):
    """
    Register a model in MLflow Model Registry and assign an alias.
    """
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        client = mlflow.tracking.MlflowClient()

        # Assign alias instead of stage
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=model_version.version
        )

        logging.info(
            "Model '%s' version %s registered with alias '%s'",
            model_name,
            model_version.version,
            alias
        )

    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info, alias="candidate")

    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        raise


if __name__ == '__main__':
    main()