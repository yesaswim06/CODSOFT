# Task 2 - Iris Flower Classification

## 🎯 Objective
Classify Iris flowers into three species — Setosa, Versicolor, Virginica — using sepal and petal measurements.

## 📊 Dataset
- Source: sklearn built-in / [UCI Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- Records: 150 samples (50 per species)
- Features: Sepal Length, Sepal Width, Petal Length, Petal Width

## 🤖 Models Used
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest Classifier

## 📈 Results
| Model | Accuracy |
|-------|----------|
| KNN | ~97% |
| SVM | ~97% |
| Random Forest | ~97% |

## 🔑 Key Insights
- Petal Length and Petal Width are the most discriminating features
- Setosa is linearly separable from the other two species
- SVM with RBF kernel performs excellently on this dataset

## ▶️ Run
```bash
python iris_classification.py
```
