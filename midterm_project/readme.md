# Cherry Leaf Disease Detection System

## Overview

This project uses machine learning and image analysis
to classify cherry leaves as healthy or infected with
powdery mildew.

The application combines exploratory data analysis,
machine learning model development, and deployment
through an interactive Streamlit dashboard.

---

## Dataset

Source:
PlantVillage Dataset

Classes:

- Healthy (854 images)
- Powdery Mildew (1052 images)

Total Images: 1906

---

## Features

The following image-derived features were extracted:

- Average Red
- Average Green
- Average Blue
- Brightness

---

## Models Evaluated

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 71.5% |
| Random Forest | 72.5% |
| Support Vector Machine | 74.9% |

The Support Vector Machine (SVM) achieved the highest performance.

---

## Streamlit Dashboard

Pages include:

- Overview
- Dataset Exploration
- Feature Analysis
- Correlation Analysis
- Model Performance
- Live Prediction
- Conclusions

---

## Running the Application
