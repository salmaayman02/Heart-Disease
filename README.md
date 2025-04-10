# 🏥 Heart Disease Risk Predictor

This project is a **Streamlit web application** that predicts whether an individual is at risk of heart disease based on health and lifestyle-related input data. The model was trained using a cleaned 2020 dataset and leverages a Random Forest Classifier for accurate predictions.

---

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
```text
- BMI
- Smoking
- AlcoholDrinking
- Stroke
- PhysicalHealth
- MentalHealth
- DiffWalking
- Sex
- AgeCategory
- Race
- Diabetic
- PhysicalActivity
- GenHealth
- SleepTime
- Asthma
- KidneyDisease
- SkinCancer
```

Target variable: `HeartDisease` (binary classification)

---

## ⚙️ How It Works

1. The user provides input values through a Streamlit form.
2. The input is encoded using previously saved `LabelEncoders` and `OneHotEncoding`.
3. The data is aligned to match the trained model's structure.
4. The Random Forest model makes a prediction on heart disease risk.
5. The prediction is displayed along with a message indicating presence or absence of risk.

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **Joblib / Pickle**

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

> Make sure you have the following files in the project directory:
> - `random_forest_model.joblib`
> - `label_encoders.pkl`
> - `encoded_columns.pkl`
> - `2020/heart_2020_cleaned.csv`

---

## 📁 Project Structure

```
├── app.py                      # Main Streamlit app
├── random_forest_model.joblib # Trained Random Forest model
├── label_encoders.pkl         # LabelEncoders for categorical features
├── encoded_columns.pkl        # Final encoded feature list
├── 2020/
│   └── heart_2020_cleaned.csv # Original dataset used for training
├── Heart Disease.ipynb        # Jupyter Notebook for training & EDA
├── heart2020.ipynb            # Additional analysis (optional)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🔮 Future Improvements

- Deploy the app on Streamlit Cloud
- Add prediction probability (`predict_proba`)
- Visualize feature importance using SHAP or Sklearn
- Save user inputs and predictions for history tracking
- Add explanations for each input field

---

## 🙌 Acknowledgments

- Data Source: [CDC BRFSS 2020](https://www.cdc.gov/brfss/)
- Inspired by the intersection of machine learning and preventive healthcare

---

