# Cherry Leaf Disease Detection Using Machine Learning

## Project Overview

Powdery mildew is a fungal disease that affects many plants and can impact plant health and productivity. 

The goal of this project was to develop a machine learning system capable of classifying leaf images, specifically cherry, as either Healthy or Powdery Mildew using image-derived features and an interactive Streamlit dashboard. 

This project works to demonstrate a complete machine learning workflow using skills developed during the course including data preparation, exploratory data analysis, model development, evaluation, and deployment. 

---

## Dataset

Source:
PlantVillage Dataset

The dataset contains: 
- 1,096 total cherry leaf images
- 854 Healthy leaf images
- 1,052 Powdery Mildew leaf images

Images were processed to extract numerical features for machine learning analysis. 

---

## Features

Each image was converted into numerical features including: 

- Average Red Intensity
- Average Green Intensity
- Average Blue Intensity
- Brightness
- Image Width
- Image Height

These features were used to train and evaluate the machine learning models. 

---

## Exploratory Data Analysis

Exploratory analysis included: 
- Summary statistics
- Histograms
- Boxplots
- Correlation analysis

These analyses revealed measurable differences between healthy and powdery mildew infected leaves. 

--- 

## Machine Learning Models

Three classification algorithms were evaluated: 

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 71.5% |
| Random Forest | 72.5% |
| Support Vector Machine | 74.9% |

The Support Vector Machine (SVM) achieved the highest classification accuracy and was selected as the final model for the midterm project checkpoint. 

### Model Evaluation
Final SVM Performance: 
- Accuracy: 74.9%
- Precision: 76.0%
- Recall: 75.0%
- F1 Score: 74.0%

---

## Streamlit Dashboard

The interactive application was built using Streamlit and includes:

- Overview Page
- Dataset Exploration
- Feature Analysis
- Correlation Analysis
- Model Performance
- Live Prediction
- Conclusions

---

## Running the Application
Clone the repository: 
```bash
git clone https://github.com/jazlynsalazar/cherry-leaf-disease-detection.git
cd cherry-leaf-disease-detection
```

Install dependencies:
```
pip install -r requirements.txt
```

Run the application:
```
streamline run app.py
```
---

## Repository Contents
- app.py : Streamlit application
- cherry_features.csv : Extracted feature dataset
- svm_model.pkl : Trained machine learning model
- notebooks/ : Analysis notebook
- images/ : Visualizations and screenshots

## Limitations
- Only color-based image features were used
- Images originated from a single dataset
- Model performance remains below 80% accuracy
- Results may not generalize to all real-world growing conditions

---

## Future Work
Potential improvements include: 
- Texture-based image features
- Deep learning models (CNNs)
- Larger and more diverse datasets
- Cloud deployment of the Streamlit application

---

## Repository Structure

Midterm_Project/
│
├── app.py
├── cherry_features.csv
├── svm_model.pkl
├── requirements.txt
├── README.md
│
├── images/
│   ├── confusion_matrix.png
│   ├── overview_page.png
│   ├── feature_analysis.png
│   ├── model_performance.png
│   └── live_prediction.png
│
├── notebooks/
│   └── Salazar_CMSE830_midterm_project.ipynb
│
└── presentation/
    └── Cherry_Leaf_Disease_Midterm.pdf

---

## AI Use Statement
Microsoft Copilot was used as a coding assistant for Streamlit implementation and debugging. All analyses, interpretations, project decisions, model evaluation, and final project content were reviewed and validated. 
