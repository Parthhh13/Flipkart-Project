# 🚀 Flipkart Customer Support CSAT Prediction (NLP + ML Capstone Project)

## 📌 Project Type
- Exploratory Data Analysis (EDA)
- Text Preprocessing (NLP)
- Classification using Machine Learning

## 👤 Contribution
**Individual Project**  
Author: Parth Pahwa

---

## 🧠 Project Summary

Flipkart, one of India's largest e-commerce platforms, receives thousands of customer service tickets daily. The quality and speed of issue resolution play a critical role in customer satisfaction, reflected in the CSAT (Customer Satisfaction) score.

This project aims to:
- Analyze historical customer support ticket data
- Preprocess structured and textual data
- Perform exploratory data analysis (EDA)
- Engineer features from raw and text data
- Train and evaluate classification models to predict CSAT score category (Satisfied / Not Satisfied)

The final ML model helps identify factors driving poor customer satisfaction and allows Flipkart to take proactive steps to improve service quality.

---

## 🎯 Business Objective

- Predict whether a customer is satisfied (CSAT ≥ 4) or not based on ticket details.
- Help Flipkart identify areas (shifts, departments, issue types) affecting satisfaction.
- Enable early interventions on at-risk tickets using predictive insights.

---

## 📂 Dataset

- **Customer Remarks** (text field)
- **Structured features** like: Channel Name, Agent Shift, Response Time, Tenure Bucket, Sub-category, etc.
- **Target**: CSAT Score (1–5), converted to binary for classification

---

## 🔍 Key Techniques

- Univariate, Bivariate, Multivariate Analysis (20+ charts)
- Text Cleaning (Lowercase, Lemmatization, Stopword Removal)
- TF-IDF Vectorization
- Label Encoding + Feature Engineering
- Outlier removal via IQR
- SMOTE for class imbalance
- Feature Selection (Chi-Square + SelectKBest)
- Dimensionality Reduction using PCA

---

## 🤖 Models Used

| Model               | Tuning             | Final Accuracy / F1 |
|---------------------|--------------------|---------------------|
| Logistic Regression | GridSearchCV       | Moderate            |
| Random Forest       | GridSearchCV       | Good                |
| XGBoost             | GridSearchCV       | ⭐Best Performer    |

Final Model: **XGBoost** with TF-IDF, PCA, and SMOTE

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score (primary metric)

---

## 🧪 Hypothesis Testing

- Sub-category significantly affects CSAT
- Agent shift affects customer satisfaction
- Longer response time negatively impacts CSAT

Statistical tests used:
- Kruskal-Wallis H-Test
- Pearson Correlation

---

## 💾 Model Deployment Ready

- Final model saved using `joblib`
- Predicts CSAT category for new tickets

---

## 📁 Folder Structure
project/
├── README.md
├── Flipkart_CSAT_EDA_and_Model.ipynb
├── csat_prediction_model.pkl
├── dataset.csv


---

## 🌐 GitHub Repository

> [Add your GitHub link here]

---

## 📈 Future Work

- Deploy model via Flask API
- Integrate with live ticket dashboard
- Build feedback loops for model retraining

---

## 📩 Contact

Parth Pahwa  
Phone: +91 7835938373
Email: parthpahwa1301@gmail.com  
LinkedIn: https://www.linkedin.com/in/parth-pahwa-4501982a8/

