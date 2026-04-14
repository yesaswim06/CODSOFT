# ============================================================
# TASK 1 - TITANIC SURVIVAL PREDICTION
# CodSoft Data Science Internship
# ============================================================
# HOW TO USE:
#   1. Download the dataset from CodSoft PDF link
#   2. Place the CSV file in THIS same folder
#   3. Run: python titanic_survival_prediction.py
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   TITANIC SURVIVAL PREDICTION - CodSoft Internship")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# Tries all common Titanic CSV filenames automatically
# ─────────────────────────────────────────────────────────────
possible_files = ['train.csv', 'titanic.csv', 'Titanic-Dataset.csv',
                  'titanic_train.csv', 'tested.csv']
csv_file = None
for f in possible_files:
    if os.path.exists(f):
        csv_file = f
        break

if csv_file:
    df = pd.read_csv(csv_file)
    print(f"\n✅ Dataset loaded from: {csv_file}")
else:
    print("\n⚠️  No CSV file found in this folder!")
    print("    Falling back to seaborn built-in Titanic dataset.")
    print("    To use the real dataset: place your CSV here and re-run.\n")
    df = sns.load_dataset('titanic')
    df.rename(columns={
        'survived':'Survived','pclass':'Pclass','sex':'Sex',
        'age':'Age','sibsp':'SibSp','parch':'Parch',
        'fare':'Fare','embarked':'Embarked'
    }, inplace=True)

# Normalize column names
df.columns = [c.strip() for c in df.columns]

print(f"\n📋 Shape     : {df.shape}")
print(f"📋 Columns   : {list(df.columns)}")
print(f"\n📋 First 5 rows:\n{df.head()}")
print(f"\n🔍 Missing Values:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 2: EDA
# ─────────────────────────────────────────────────────────────
# Detect survived column name
surv_col = 'Survived' if 'Survived' in df.columns else 'survived'

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Titanic EDA - CodSoft Internship', fontsize=16, fontweight='bold')

sex_col    = next((c for c in df.columns if c.lower()=='sex'), None)
pclass_col = next((c for c in df.columns if c.lower()=='pclass'), None)
age_col    = next((c for c in df.columns if c.lower()=='age'), None)
fare_col   = next((c for c in df.columns if c.lower()=='fare'), None)

sns.countplot(x=surv_col, data=df, palette=['#e74c3c','#2ecc71'], ax=axes[0,0])
axes[0,0].set_title('Survival Count (0=No, 1=Yes)')

if sex_col:
    sns.countplot(x=sex_col, hue=surv_col, data=df,
                  palette=['#e74c3c','#2ecc71'], ax=axes[0,1])
    axes[0,1].set_title('Survival by Gender')

if pclass_col:
    sns.countplot(x=pclass_col, hue=surv_col, data=df,
                  palette=['#e74c3c','#2ecc71'], ax=axes[0,2])
    axes[0,2].set_title('Survival by Passenger Class')

if age_col:
    axes[1,0].hist(df[df[surv_col]==1][age_col].dropna(), bins=20,
                   alpha=0.7, color='#2ecc71', label='Survived')
    axes[1,0].hist(df[df[surv_col]==0][age_col].dropna(), bins=20,
                   alpha=0.7, color='#e74c3c', label='Did Not Survive')
    axes[1,0].set_title('Age Distribution')
    axes[1,0].legend()

if fare_col:
    axes[1,1].hist(df[df[surv_col]==1][fare_col].dropna(), bins=30,
                   alpha=0.7, color='#2ecc71', label='Survived')
    axes[1,1].hist(df[df[surv_col]==0][fare_col].dropna(), bins=30,
                   alpha=0.7, color='#e74c3c', label='Did Not Survive')
    axes[1,1].set_title('Fare Distribution')
    axes[1,1].legend()

if pclass_col:
    rate = df.groupby(pclass_col)[surv_col].mean()
    axes[1,2].bar(rate.index, rate.values, color=['#3498db','#9b59b6','#e67e22'])
    axes[1,2].set_title('Survival Rate by Class')
    axes[1,2].set_ylabel('Survival Rate')

plt.tight_layout()
plt.savefig('titanic_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n📈 EDA saved: titanic_eda.png")

# ─────────────────────────────────────────────────────────────
# STEP 3: PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("\n🔧 Preprocessing...")

# Pick available features from the dataset
candidate_features = {
    'pclass':   next((c for c in df.columns if c.lower()=='pclass'),   None),
    'sex':      next((c for c in df.columns if c.lower()=='sex'),      None),
    'age':      next((c for c in df.columns if c.lower()=='age'),      None),
    'sibsp':    next((c for c in df.columns if c.lower()=='sibsp'),    None),
    'parch':    next((c for c in df.columns if c.lower()=='parch'),    None),
    'fare':     next((c for c in df.columns if c.lower()=='fare'),     None),
    'embarked': next((c for c in df.columns if c.lower()=='embarked'), None),
}
features = [v for v in candidate_features.values() if v is not None]
df_model = df[features + [surv_col]].copy()

# Fill missing values
for col in features:
    if df_model[col].dtype in [np.float64, np.int64]:
        df_model[col].fillna(df_model[col].median(), inplace=True)
    else:
        df_model[col].fillna(df_model[col].mode()[0], inplace=True)

# Encode categorical
le = LabelEncoder()
for col in features:
    if df_model[col].dtype == object:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

print(f"✅ Missing values remaining: {df_model.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 4: SPLIT & TRAIN
# ─────────────────────────────────────────────────────────────
X = df_model[features]
y = df_model[surv_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n📦 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print(f"\n📊 Logistic Regression : {lr_acc*100:.2f}%")
print(f"📊 Random Forest       : {rf_acc*100:.2f}%")
print(f"\n🏆 Best: Random Forest ({rf_acc*100:.2f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 5: EVALUATION
# ─────────────────────────────────────────────────────────────
print(f"\n📋 Classification Report:\n")
print(classification_report(y_test, rf_preds,
                             target_names=['Not Survived','Survived']))

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Titanic - Model Evaluation', fontsize=14, fontweight='bold')

ConfusionMatrixDisplay(confusion_matrix(y_test, rf_preds),
    display_labels=['Not Survived','Survived']).plot(
    ax=axes2[0], colorbar=False, cmap='Blues')
axes2[0].set_title('Confusion Matrix')

feat_df = pd.DataFrame({'Feature': features,
                        'Importance': rf.feature_importances_})
feat_df = feat_df.sort_values('Importance', ascending=True)
axes2[1].barh(feat_df['Feature'], feat_df['Importance'], color='#3498db')
axes2[1].set_title('Feature Importance')

plt.tight_layout()
plt.savefig('titanic_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n📈 Results saved: titanic_results.png")
print("\n" + "="*60)
print("   TASK 1 COMPLETE ✅")
print("="*60)
