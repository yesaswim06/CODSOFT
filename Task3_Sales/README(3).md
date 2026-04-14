# Task 3 - Sales Prediction Using Python

## 🎯 Objective
Predict product sales based on advertising spend across TV, Radio, and Newspaper channels.

## 📊 Dataset
- Source: Advertising dataset (advertising.csv)
- Records: 200 campaigns
- Features: TV budget, Radio budget, Newspaper budget → Sales

## 🤖 Models Used
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor *(Best)*

## 📈 Results
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | ~1.5 | ~2.0 | ~0.89 |
| Random Forest | ~0.8 | ~1.1 | ~0.95 |
| Gradient Boosting | ~0.75 | ~1.0 | ~0.96 |

## 🔑 Key Insights
- TV advertising has the strongest correlation with sales
- Radio also contributes meaningfully
- Newspaper advertising has minimal impact on sales

## ▶️ Run
```bash
python sales_prediction.py
```
