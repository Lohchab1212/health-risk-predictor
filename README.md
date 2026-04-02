# Health Risk Predictor 
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Sklearn](https://img.shields.io/badge/ScikitLearn-1.x-orange)
![SMOTE](https://img.shields.io/badge/SMOTE-Imbalanced--Learn-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
## Overview
This project aims to analyze health data and predict whether a person is likely to have diabetes.
It is built as a learning project to understand the complete ML workflow which is from data exploration to model building.
## Installation
```bash
Clone the repository
git clone https://github.com/Lohchab1212/health-risk-predictor.git
cd health-risk-predictor
```
```bash
Install required libraries
ip install pandas matplotlib seaborn scikit-learn
```
```bash
Open the notebook health_risk.ipynb in Jupyter or VS Code and run all cells.
```
## Dataset 
Diabetes Health Indicators Dataset
253,680 survey responses from cleaned BRFSS 2015
## Problem -Class Imbalance 
While EDA i saw there is an imbalance in the dataset. There are more patients with no diabtetes and very less patients who are pre diabeteic or diabetic which eventually ill lead our model to give false 
predictions even if our model prediction. If we train a model on this data as it is, it could just predict "no diabetes" for every single person and still be 84% accurate because 84% of the dataset actually has no diabetes. But that model is completely useless in real life because it never catches a single diabetic patient!
In healthcare especially, this is dangerous. Missing a diabetic patient is far worse than a false alarm.
## How we Fixed the problem
How we fix it, 3 common ways:
- Undersampling — reduce the majority class (remove some 0s)
- Oversampling — increase the minority class (duplicate some 1s)
- SMOTE — synthetically generate new minority class samples
We'll use SMOTE because it's the most sophisticated and widely used.
The key reason we apply SMOTE only on training data is:
We never want to touch the test data. The test sset represents the real world. If we apply SMOTE on the full dataset before splitting, synthetic samples could leak iinto the test set and we would be 
evaluating out model on the fake data. that gives us false results that would never work in real life scenario.
## Model Performance
## 📊 Model Performance
### Confusion Matrix
- True Positives (Diabetes correctly predicted): **6091**  
- True Negatives (No Diabetes correctly predicted): **30930**  
- False Positives: **11865**  
- False Negatives: **1850**  
### 📈 ROC Curve
- **AUC Score: ~0.82**
### 💡 Interpretation
- Model performs well in distinguishing between classes  
- Low false negatives → important for healthcare applications  
- Some false positives exist → scope for improvement  
---
## 🧠 Key Observations
- This Model prioritizes minimising false negatives, as missing a diabetic case can have a serious real life consequences.
- BMI shows higher values in diabetic individuals  
- Dataset imbalance required correction (handled using SMOTE)  
- Logistic Regression provides a strong baseline  
## Tools Used
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Imbalanced-learn (SMOTE)
## Next steps
- Try advanced models (Random Forest, XGBoost)
- Hyperparameter tuning
- Build a web app using Flask/Streamlit
- Deploy the model
## Hyperparameter Tuning
Used GridSearchCV with 5-fold cross validation across 30 
parameter combinations.

Best Parameters:
- C: 0.1
- max_iter: 100  
- solver: liblinear

Final ROC AUC after tuning: 0.822

## Model Comparison
| Model | CV AUC | Test AUC | Verdict |
|---|---|---|---|
| Logistic Regression | 0.823 | 0.822 | ✅ Best |
| XGBoost | 0.960 | 0.820 | ❌ Overfitting |
| Random Forest | 0.975 | - | ❌ Overfitting |

Logistic Regression selected as final model due to 
consistency and reliability over more complex models.
Despite XGBoost showing 0.96 AUC during cross validation, test evaluation revealed overfitting (0.82). Logistic Regression proved most reliable with consistent CV and test AUC of 0.822 — demonstrating that model complexity doesn't always mean better performance.
## Deployment
The model is deployed as a REST API using FastAPI.

- Framework: FastAPI
- Server: Uvicorn
- Endpoint: POST /predict
- Input: 21 health indicators
- Output: Risk prediction + probability score

### Sample Response
```json
{
  "prediction": 1,
  "risk": "High Risk", 
  "probability": 0.893,
  "message": "Please consult a doctor for proper diagnosis."
}
```
## Limitations
- Dataset skews toward older population (mean age 63)
- Model may overestimate risk for younger individuals
- Not intended for medical diagnosis

## ⚠️ Disclaimer
This project is for **educational purposes only** and should not be used for medical diagnosis.
---
## 🙌 Author
**Vishal Lohchab**
---
⭐ If you like this project, consider giving it a star!
