# ============================================================
# TASK 3 - IRIS FLOWER CLASSIFICATION
# CodSoft Data Science Internship
# ============================================================
# HOW TO USE:
#   1. Download the dataset from CodSoft PDF link
#   2. Place the CSV file in THIS same folder
#   3. Run: python iris_classification.py
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   IRIS FLOWER CLASSIFICATION - CodSoft Internship")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
possible_files = ['IRIS.csv', 'iris.csv', 'Iris.csv',
                  'iris_dataset.csv', 'iris_data.csv']
csv_file = None
for f in possible_files:
    if os.path.exists(f):
        csv_file = f
        break

species_names = ['setosa', 'versicolor', 'virginica']

if csv_file:
    df = pd.read_csv(csv_file)
    print(f"\n✅ Dataset loaded from: {csv_file}")
    print(f"\n📋 Columns: {list(df.columns)}")

    # Detect species column
    species_col = next((c for c in df.columns
                        if c.lower() in ['species','class','variety']), None)
    if species_col is None:
        species_col = df.columns[-1]  # assume last column is target

    feature_cols = [c for c in df.columns if c != species_col]

    le = LabelEncoder()
    df['target'] = le.fit_transform(df[species_col].astype(str))
    species_names = list(le.classes_)

    X = df[feature_cols].values
    y = df['target'].values

else:
    print("\n⚠️  No CSV found. Using sklearn built-in Iris dataset.")
    print("    Place IRIS.csv here to use the real dataset.\n")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_cols = list(iris.feature_names)
    species_names = list(iris.target_names)
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    df['species'] = [species_names[i] for i in y]

print(f"\n📋 Shape  : {df.shape}")
print(f"📋 Species: {species_names}")
print(f"\n📋 First 5 rows:\n{df.head()}")
print(f"\n📊 Class Distribution:")
target_col = 'species' if 'species' in df.columns else 'target'
print(df[target_col].value_counts())

# ─────────────────────────────────────────────────────────────
# STEP 2: EDA
# ─────────────────────────────────────────────────────────────
colors = ['#e74c3c', '#3498db', '#2ecc71']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Iris EDA - CodSoft Internship', fontsize=16, fontweight='bold')

pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
for idx, (i, j) in enumerate(pairs):
    if i >= len(feature_cols) or j >= len(feature_cols):
        continue
    row, col = divmod(idx, 3)
    for sp_idx, sp_name in enumerate(species_names):
        mask = y == sp_idx
        axes[row, col].scatter(X[mask, i], X[mask, j],
                               c=colors[sp_idx % 3], label=sp_name,
                               alpha=0.7, s=40)
    axes[row, col].set_xlabel(str(feature_cols[i]).replace(' (cm)',''))
    axes[row, col].set_ylabel(str(feature_cols[j]).replace(' (cm)',''))
    axes[row, col].legend(fontsize=7)

plt.tight_layout()
plt.savefig('iris_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n📈 EDA saved: iris_eda.png")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Iris Correlation & Distribution', fontsize=14, fontweight='bold')

feat_df = pd.DataFrame(X, columns=feature_cols)
sns.heatmap(feat_df.corr(), annot=True, fmt='.2f',
            cmap='coolwarm', ax=axes2[0], square=True)
axes2[0].set_title('Feature Correlation')

feat_df['species'] = [species_names[i] for i in y]
feat_melt = feat_df.melt(id_vars='species', var_name='Feature', value_name='Value')
sns.boxplot(x='Feature', y='Value', hue='species', data=feat_melt,
            palette=['#e74c3c','#3498db','#2ecc71'], ax=axes2[1])
axes2[1].set_title('Feature Distribution by Species')
axes2[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('iris_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────
# STEP 3: PREPROCESSING & SPLIT
# ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📦 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────
# STEP 4: TRAIN MODELS
# ─────────────────────────────────────────────────────────────
print("\n🤖 Training models...")

models = {
    'K-Nearest Neighbors':    KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', C=1.0, random_state=42),
    'Random Forest':          RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv  = cross_val_score(model, X_scaled, y, cv=5)
    results[name] = {'model': model, 'preds': preds, 'accuracy': acc,
                     'cv_mean': cv.mean(), 'cv_std': cv.std()}
    print(f"\n   {name}:")
    print(f"     Test Accuracy : {acc*100:.2f}%")
    print(f"     CV Accuracy   : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

best_name = max(results, key=lambda k: results[k]['accuracy'])
best = results[best_name]
print(f"\n🏆 Best: {best_name} ({best['accuracy']*100:.2f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 5: EVALUATION
# ─────────────────────────────────────────────────────────────
print(f"\n📋 Classification Report ({best_name}):\n")
print(classification_report(y_test, best['preds'], target_names=species_names))

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle(f'Iris - Model Evaluation ({best_name})', fontsize=14, fontweight='bold')

names = list(results.keys())
accs  = [results[n]['accuracy']*100 for n in names]
bar_colors = ['#2ecc71' if n == best_name else '#3498db' for n in names]
bars = axes3[0].bar(names, accs, color=bar_colors, width=0.5)
axes3[0].set_ylim(85, 105)
axes3[0].set_title('Model Accuracy Comparison')
axes3[0].set_ylabel('Accuracy (%)')
for bar, acc in zip(bars, accs):
    axes3[0].text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.3, f'{acc:.1f}%',
                  ha='center', fontweight='bold')
axes3[0].tick_params(axis='x', rotation=15)

ConfusionMatrixDisplay(confusion_matrix(y_test, best['preds']),
    display_labels=species_names).plot(
    ax=axes3[1], colorbar=False, cmap='Blues')
axes3[1].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('iris_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n📈 Results saved: iris_results.png")

# ─────────────────────────────────────────────────────────────
# STEP 6: SAMPLE PREDICTIONS
# ─────────────────────────────────────────────────────────────
print("\n🔮 Sample Predictions:")
print("-" * 55)
samples = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.7, 3.1, 5.6, 2.4],
])
samples_scaled = scaler.transform(samples)
preds = best['model'].predict(samples_scaled)
print(f"  {'F1':>5} {'F2':>5} {'F3':>5} {'F4':>5}  →  Species")
print("  " + "-" * 40)
for s, p in zip(samples, preds):
    print(f"  {s[0]:>5.1f} {s[1]:>5.1f} {s[2]:>5.1f} {s[3]:>5.1f}"
          f"  →  🌸 {species_names[p].upper()}")

print("\n" + "="*60)
print("   TASK 2 COMPLETE ✅")
print("="*60)
