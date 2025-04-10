# 🏥 Heart Disease Risk Predictor

This project is a **Streamlit web application** that predicts whether an individual is at risk of heart disease based on health and lifestyle-related input data. The model was trained using a cleaned 2020 dataset and leverages a Random Forest Classifier for accurate predictions.



## 🧠 Project Overview

### Features:
- User-friendly interface built with **Streamlit**
- Predicts risk based on 17+ features like BMI, Smoking, Stroke, Sleep, Age, and more
- Built-in **data preprocessing** (label encoding & one-hot encoding)
- **Random Forest Classifier** model trained on 300K+ samples
- Handles missing columns automatically using saved `encoded_columns.pkl`

---

## 📊 Dataset

- 📁 **Source**: CDC Behavioral Risk Factor Surveillance System (BRFSS)
- 🧼 **Cleaned CSV**: `heart_2020_cleaned.csv` (~320,000 rows)

### Input Features:
