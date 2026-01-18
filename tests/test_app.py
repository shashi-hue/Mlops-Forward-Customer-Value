import unittest
import json
from flask_app.app import app


class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        app.testing = True
        cls.client = app.test_client()


    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<html", response.data)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "ok")


    def test_predict_api_success(self):
        payload = [
            {
                "unique_invoices": 5,
                "total_quantity": 100,
                "avg_quantity_per_order": 20.0,
                "unit_price_std": 10.5,
                "customer_age_days": 365,
                "days_since_last_purchase": 30,
                "average_days_between_purchase": 45.0,
                "is_onetime_buyer": 0,
            }
        ]

        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("predictions", data)
        self.assertEqual(len(data["predictions"]), 1)
        self.assertIsInstance(data["predictions"][0], float)

    def test_predict_api_missing_features(self):
        payload = [
            {
                "unique_invoices": 5,
                "total_quantity": 100
                # missing remaining features
            }
        ]

        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)



    def test_predict_form_success(self):
        form_data = {
            "unique_invoices": "5",
            "total_quantity": "100",
            "avg_quantity_per_order": "20.0",
            "unit_price_std": "10.5",
            "customer_age_days": "365",
            "days_since_last_purchase": "30",
            "average_days_between_purchase": "45.0",
            "is_onetime_buyer": "0",
        }

        response = self.client.post("/predict-form", data=form_data)
        self.assertEqual(response.status_code, 200)

        self.assertIn(b"Predicted", response.data)


    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"app_request_count", response.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
