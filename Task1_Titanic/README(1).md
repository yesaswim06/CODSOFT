# Task 1 - Titanic Survival Prediction

## 🎯 Objective
Predict whether a Titanic passenger survived based on features like age, gender, and ticket class.

## 📊 Dataset
- Source: Seaborn built-in / [Kaggle Titanic](https://www.kaggle.com/c/titanic/data)
- Records: 891 passengers
- Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

## 🤖 Models Used
- Logistic Regression
- Random Forest Classifier *(Best)*

## 📈 Results
| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~80% |
| Random Forest | ~82% |

## 🔑 Key Insights
- Female passengers had much higher survival rates than males
- 1st class passengers survived at higher rates than 2nd/3rd
- Age and Fare were important predictors

## ▶️ Run
```bash
python titanic_survival_prediction.py
```
