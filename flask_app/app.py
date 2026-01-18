import numpy as np
import pandas as pd
import mlflow
import dagshub
import os
from flask import Flask, render_template, request, jsonify
import time
from prometheus_client import Counter, Histogram, CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest 

# For local use
# dagshub.init(repo_owner='shashi-hue', repo_name='Mlops-Forward-Customer-Value', mlflow=True)

#For production
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


app = Flask(__name__)


# Defining custom metrics
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "total number of requests to the app", ["method", "endpoint"], registry=registry)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry)


# Model setup
model_name = "my_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None


model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)


REQUIRED_FEATURES = [
    "unique_invoices",
    "total_quantity",
    "avg_quantity_per_order",
    "unit_price_std",
    "customer_age_days",
    "days_since_last_purchase",
    "average_days_between_purchase",
    "is_onetime_buyer",
]


@app.route("/", methods=["GET"])
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.route("/predict-form", methods=["POST"])
def predict_form():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict-form").inc()
    start_time = time.time()
    try:
        data = {
            "unique_invoices": int(request.form["unique_invoices"]),
            "total_quantity": int(request.form["total_quantity"]),
            "avg_quantity_per_order": float(request.form["avg_quantity_per_order"]),
            "unit_price_std": float(request.form["unit_price_std"]),
            "customer_age_days": int(request.form["customer_age_days"]),
            "days_since_last_purchase": int(request.form["days_since_last_purchase"]),
            "average_days_between_purchase": float(request.form["average_days_between_purchase"]),
            "is_onetime_buyer": int(request.form["is_onetime_buyer"]),
        }

        df = pd.DataFrame([data])
        preds_log = model.predict(df)
        prediction = float(np.expm1(preds_log)[0])

        REQUEST_LATENCY.labels(endpoint="/predict-form").observe(time.time() - start_time)

        return render_template(
            "index.html",
            prediction=round(prediction, 2)
        )

    except Exception as e:
        REQUEST_LATENCY.labels(endpoint="/predict-form").observe(time.time() - start_time)
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )


@app.route("/predict", methods=["POST"])
def predict_api():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    try:
        payload = request.get_json()
        df = pd.DataFrame(payload)

        missing = set(REQUIRED_FEATURES) - set(df.columns)
        if missing:
            REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
            return jsonify(
                {"error": f"Missing required features: {missing}"}
            ), 400

        X = df[REQUIRED_FEATURES]
        preds_log = model.predict(X)
        preds = np.expm1(preds_log)

        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose Prometheus metrics."""
    REQUEST_COUNT.labels(method="GET", endpoint="/metrics").inc()
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)