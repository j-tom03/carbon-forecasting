# Carbon Intensity Forecasting: Temporal Fusion Transformer
A production-grade MLOps pipeline for forecasting grid carbon intensity (gCO₂/kWh) 96-hours ahead using a Temporal Fusion Transformer (TFT). This project demonstrates an end-to-end machine learning lifecycle, from data ingestion and versioning to model deployment and monitoring.


## Model Architecture

The forecasting model is based on a Temporal Fusion Transformer (TFT), selected for its ability to model multivariate time series with static and time-varying covariates, while producing probabilistic forecasts.

Detailed architectural choices and training configuration are documented once the model pipeline is finalised.


#### Design Principles
- Reproducible experiments over ad-hoc notebooks
- Automation over manual intervention
- Production-aware ML system design



## Installation
1. Clone the repository

```bash
git clone https://github.com/j-tom03/carbon-forecasting.git
cd carbon-forecasting
```

2. Install dependencies with uv (install [here](https://docs.astral.sh/uv/getting-started/installation/))
```bash
uv sync
```

## Dependencies
This project relies on a modern Python ML stack, including:
- Deep learning and time-series modelling
- Experiment tracking and model registry
- Hyperparameter optimisation
- API serving and orchestration

A full dependency list is defined in `pyproject.toml`.

#### Hardware Requirements
- CPU-only training supported
- GPU optional for faster experimentation

## Training & Experiments
Model training is fully automated and reproducible, with experiments tracked via an experiment registry. Hyperparameters are selected using automated optimisation, and the best-performing model is promoted based on validation metrics.

Implementation details and experiment results will be added once the training pipeline is complete.


## Deployment
The trained model is exposed via a RESTful inference API, designed for low-latency probabilistic forecasts. The service supports versioned models and is deployable using containerisation on free-tier cloud infrastructure.

Deployment instructions and API documentation will be added once the service is live.
