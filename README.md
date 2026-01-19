# Forward 90-Day Customer Value — MLOps Project

This repository contains an end-to-end MLOps implementation for predicting **customer spending over the next 90 days**.

The goal of the project is to demonstrate how an experimental machine learning model can be developed, validated, deployed, and monitored as a **production-ready service**.

Rather than focusing only on model performance, the project emphasizes:
- Reproducibility
- Automation
- Controlled model promotion
- Scalable deployment
- Production observability

**MLflow tracking & model registry (DagsHub):**  
https://dagshub.com/shashi-hue/Mlops-Forward-Customer-Value.mlflow

---

## Problem Overview

The task is a regression problem: given historical customer transaction data, estimate the **total value a customer is expected to generate in the next 90 days**.

The trained model is exposed via a REST API and deployed to Kubernetes, with metrics collected continuously in production.

---

## System Overview

At a high level, the system is organized into four layers:

### 1. Experimentation & Training
- Notebooks for exploration and feature analysis
- MLflow for experiment tracking and model registry

### 2. Reproducible Pipelines
- DVC for data versioning and pipeline orchestration
- Parameterized pipelines to ensure repeatable results

### 3. Deployment & Serving
- Flask application for inference and health checks
- Docker images built and promoted via CI
- Kubernetes (EKS) for scalable serving

### 4. Monitoring
- Prometheus for metrics scraping
- Grafana dashboards for visibility into system health

---

## Repository Structure

The project follows a cookiecutter-style data science layout, adapted for production workflows:

```
.
├── .dvc/
├── .github/
│   └── workflows/
│       └── ci.yaml
├── docs/
├── flask_app/
│   ├── templates/
│   │   └── index.html
│   ├── app.py
│   └── requirements.txt
├── models/
├── notebooks/
│   ├── .gitkeep
│   ├── Model_experimentation.ipynb
│   └── retail-data.csv
├── references/
├── reports/
│   ├── figures/
│   ├── .gitkeep
│   └── experiment_info.json
├── scripts/
│   └── promote_model.py
├── src/
│   ├── connections/
│   │   └── s3_connection.py
│   ├── data/
│   │   ├── .gitkeep
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py
│   ├── features/
│   │   ├── .gitkeep
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── logger/
│   │   └── __init__.py
│   ├── model/
│   │   ├── .gitkeep
│   │   ├── __init__.py
│   │   ├── model_building.py
│   │   ├── model_evaluation.py
│   │   └── register_model.py
│   ├── visualization/
│   ├── __init__.py
│   └── utils.py
├── tests/
│   ├── test_app.py
│   └── test_model.py
├── .dvcignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── deployment.yaml
├── dvc.lock
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── setup.py
├── test_environment.py
└── tox.ini
```

Each stage of the ML lifecycle is explicitly represented in code, which makes the workflow easier to audit, test, and extend.

---

## Experimentation and Tracking

Exploration and model development are done in `notebooks/model_experimentation.ipynb`.

All experiments log parameters, metrics, and artifacts to a **remote MLflow server**.

This setup allows:
- Comparison of model versions
- Consistent evaluation across runs
- Traceability from production models back to training data and code

Models that meet validation criteria are registered in the MLflow Model Registry.

---

## Reproducible Training with DVC

Training is formalized using **DVC pipelines**, with data stored remotely in S3.

Pipeline stages include:
1. Data ingestion
2. Preprocessing
3. Feature engineering
4. Model training
5. Model evaluation

Running the full pipeline locally or in CI is as simple as:

```bash
dvc pull
dvc repro
```

This ensures that results are deterministic and reproducible across environments.

---

## CI/CD and Model Governance

The CI workflow defined in `.github/workflows/ci.yaml` automates:

- Environment setup
- Pipeline execution
- Unit and integration tests
- Model evaluation checks
- Conditional model registration and promotion
- Docker image build and push
- Deployment steps

Only models that pass predefined checks are eligible for promotion, reducing the risk of manual errors.

---

## Model Serving

The trained model is served through a Flask REST API that provides:

- Prediction endpoints
- Health checks for orchestration and monitoring

The service is containerized with Docker and configured via environment variables to keep code and infrastructure concerns separate.

---

## Kubernetes Deployment

The application is deployed to a managed Kubernetes cluster (EKS).

Key characteristics:

- Container images built and promoted via CI
- Rolling updates for zero-downtime deployments
- External access through a LoadBalancer service

This setup mirrors common production deployment patterns for ML services.

---

## Monitoring and Observability

Once deployed, the service exposes metrics that are scraped by Prometheus.

Grafana dashboards provide visibility into:

- Request volume
- Response latency
- Error rates
- Service availability

This allows operational issues to be detected early and tied back to model or infrastructure changes.

---

## Why This Project Exists

This repository is intended to demonstrate:

- How to move from notebooks to production systems
- How to combine MLflow and DVC for reproducible ML
- How CI/CD can enforce model quality gates
- How deployed models can be monitored like any other service

The emphasis is on engineering discipline, not just experimentation.

---

## How to Explore the Project

### View experiments and registered models
**MLflow UI:**  
https://dagshub.com/shashi-hue/Mlops-Forward-Customer-Value.mlflow

### Review experimentation and feature work
`notebooks/model_experimentation.ipynb`

### Inspect training pipelines
`dvc.yaml`, `params.yaml`

### Review CI/CD logic
`.github/workflows/ci.yaml`

### Inspect serving code
`flask_app/`

---

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/shashi-hue/Mlops-Forward-Customer-Value.git
   cd Mlops-Forward-Customer-Value
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull data and run pipeline**
   ```bash
   dvc pull
   dvc repro
   ```

4. **Run the Flask app locally**
   ```bash
   cd flask_app
   python app.py
   ```

---

## License

MIT License - see LICENSE file for details.
