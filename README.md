# 🇳🇬 Nigeria Crop Yield Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nigeria-crop-yield-predictor.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)

**Live Demo:** [Click here to test the app](https://nigeria-crop-yield-predictor.streamlit.app/)

## 📖 Project Overview
This project is a machine learning solution designed to support Precision Agriculture in Nigeria. It predicts the expected crop yield (kg/hectare) for major Nigerian crops based on environmental factors (rainfall, temperature) and farm size.

Unlike standard datasets, the data for this project was **custom-engineered** by merging agricultural output data from the **FAO (Food and Agriculture Organization)** with historical climate data from the **World Bank/NIMET**, covering the years 1990–2024.

## 🚀 Key Features
* **Custom Data Pipeline:** Scripts to scrape, clean, pivot, and merge disparate data sources into a usable ML dataset.
* **High-Performance Model:** Trained an **XGBoost Regressor** achieving an **R² Score of ~0.98** on test data.
* **Interactive Dashboard:** A Streamlit frontend allowing farmers and policymakers to simulate climate scenarios.
* **Robust Preprocessing:** Automated handling of categorical crop data using Scikit-Learn Pipelines.

## 🛠️ Tech Stack
* **Language:** Python
* **Modeling:** XGBoost (Gradient Boosting), Scikit-Learn
* **Data Engineering:** Pandas, NumPy
* **Deployment:** Streamlit Cloud
* **Serialization:** Joblib

## 📂 Project Structure
```text
├── data/                        # Raw FAO data and processed CSVs
├── models/                      # Serialized model (.pkl) artifacts
├── notebooks/                   # Jupyter notebooks for EDA and experimentation
├── streamlit_app.py             # The frontend application
├── train_model.py               # Reproducible training pipeline script
├── requirements.txt             # Project dependencies
└── README.md                    # Documentation