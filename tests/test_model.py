import unittest
import mlflow
import os
import pandas as pd
import numpy as np
from src.utils import evaluate_regression, inverse_rmse, spearman_rank

class TestModelLoading(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "shashi-hue"
        repo_name = "Mlops-Forward-Customer-Value"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    
        
        
        cls.REQUIRED_FEATURES = [
                "unique_invoices", "total_quantity", "avg_quantity_per_order",
                "unit_price_std", "customer_age_days", "days_since_last_purchase",
                "average_days_between_purchase", "is_onetime_buyer"
            ]
        

    @staticmethod
    def get_latest_model_version(model_name, stage="None"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None
    

    def test_model_loaded_properly(self):
        """Test that model loads without errors."""
        self.assertIsNotNone(self.new_model)
        print(f"Model loaded successfully from: {self.new_model_uri}")

    def test_model_signature(self):
        """Test model input/output signature with dummy data."""
        
        dummy_data = {
            "unique_invoices": [5],
            "total_quantity": [100],
            "avg_quantity_per_order": [20.0],
            "unit_price_std": [10.5],
            "customer_age_days": [365],
            "days_since_last_purchase": [30],
            "average_days_between_purchase": [45.0],
            "is_onetime_buyer": [0]
        }
        input_df = pd.DataFrame(dummy_data)
        
        # Predict and verify shapes
        prediction = self.new_model.predict(input_df)
        print(f"Input shape: {input_df.shape}, Output: {prediction.shape}")
        
        self.assertEqual(input_df.shape[0], len(prediction))
        self.assertEqual(prediction.shape, (1,)) 

    def test_model_performance(self):
        """Test regression performance on holdout data."""
        # Extract features and target (log-scale expected)
        X_holdout = self.holdout_data[self.REQUIRED_FEATURES]
        y_holdout = self.holdout_data['target_clv']  # Your target column
        
        print(f"Test set: {X_holdout.shape[0]} samples")
        
        # Predict (log-scale output)
        y_pred = self.new_model.predict(X_holdout)
        
        log_metrics = evaluate_regression(y_holdout, y_pred)

        # --- Real-scale RMSE ---
        rmse_real = inverse_rmse(y_holdout, y_pred)

        spearman = spearman_rank(y_holdout, y_pred)

        # Debug output
        print(
            f"Log RMSE: {log_metrics['rmse']:.3f}, "
            f"Log MAE: {log_metrics['mae']:.3f}, "
            f"R²: {log_metrics['r2']:.3f}, "
            f"Real RMSE: ${rmse_real:.2f}, "
            f"Spearman: {spearman:.3f}"
        )


        self.assertLessEqual(log_metrics["rmse_log"], 0.8, "Log RMSE too high")
        self.assertLessEqual(rmse_real, 1400.0, "Real RMSE exceeds 1400")
        self.assertGreaterEqual(log_metrics["r2"], 0.4, "R² below acceptable threshold")
        self.assertGreaterEqual(spearman, 0.5, "Poor rank correlation")

    def test_feature_validation(self):
        """Test model handles missing/wrong features."""
        # Missing feature test
        bad_data = self.holdout_data[self.REQUIRED_FEATURES].copy()
        bad_data.drop(columns=['unique_invoices'], inplace=True)
        
        with self.assertRaises(Exception):
            self.new_model.predict(bad_data)
        print("Missing feature handling OK")

    def test_data_quality(self):
        """Test input data quality checks."""
        # NaN values
        data_with_nan = self.holdout_data[self.REQUIRED_FEATURES].copy()
        data_with_nan.iloc[0, 0] = np.nan
        
        with self.assertRaises(Exception):
            self.new_model.predict(data_with_nan)
        print("NaN handling OK")

if __name__ == "__main__":
    unittest.main(verbosity=2)
