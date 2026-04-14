# ============================================================
# TASK 4 - SALES PREDICTION USING PYTHON
# CodSoft Data Science Internship
# ============================================================
# HOW TO USE:
#   1. Download the dataset from CodSoft PDF link
#   2. Place the CSV file in THIS same folder
#   3. Run: python sales_prediction.py
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   SALES PREDICTION USING PYTHON - CodSoft Internship")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
possible_files = [
    'advertising.csv', 'Advertising.csv', 'sales.csv',
    'Sales.csv', 'sales_data.csv', 'advertising_data.csv'
]
csv_file = None
for f in possible_files:
    if os.path.exists(f):
        csv_file = f
        break

if csv_file:
    df = pd.read_csv(csv_file)
    print(f"\n✅ Dataset loaded from: {csv_file}")
else:
    print("\n⚠️  No CSV found. Generating sample advertising dataset.")
    print("    Place advertising.csv here to use the real dataset.\n")
    np.random.seed(42)
    n = 200
    TV        = np.random.uniform(0.7, 296.4, n)
    Radio     = np.random.uniform(0.0, 49.6,  n)
    Newspaper = np.random.uniform(0.3, 114.0, n)
    Sales     = (0.047*TV + 0.178*Radio + 0.001*Newspaper
                 + np.random.normal(0, 1.5, n) + 2.9)
    Sales     = np.clip(Sales, 1.6, 27.0)
    df = pd.DataFrame({'TV': TV, 'Radio': Radio,
                       'Newspaper': Newspaper, 'Sales': Sales})

# Normalize column names
df.columns = [c.strip() for c in df.columns]

print(f"\n📋 Shape  : {df.shape}")
print(f"📋 Columns: {list(df.columns)}")
print(f"\n📋 First 5 rows:\n{df.head()}")
print(f"\n📊 Statistical Summary:\n{df.describe()}")

# ─────────────────────────────────────────────────────────────
# STEP 2: AUTO-DETECT COLUMNS
# Works with both Advertising.csv and other sales CSVs
# ─────────────────────────────────────────────────────────────
# Try to find the target column (Sales / sales / Revenue etc.)
target_col = next((c for c in df.columns
                   if c.lower() in ['sales','revenue','target','units']), None)
if target_col is None:
    target_col = df.columns[-1]   # fallback: last column

# Feature columns = everything except target
feature_cols = [c for c in df.columns if c != target_col]

# Drop non-numeric columns
df_num = df[feature_cols + [target_col]].select_dtypes(include=[np.number])
feature_cols = [c for c in df_num.columns if c != target_col]

print(f"\n🎯 Target   : {target_col}")
print(f"📥 Features : {feature_cols}")

# Drop rows with missing values
df_num.dropna(inplace=True)
print(f"📦 Clean rows: {len(df_num)}")

# ─────────────────────────────────────────────────────────────
# STEP 3: EDA
# ─────────────────────────────────────────────────────────────
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#e67e22']

n_features = len(feature_cols)
n_cols = min(n_features, 3)
n_rows = (n_features + n_cols - 1) // n_cols + 1

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
fig.suptitle('Sales Prediction EDA - CodSoft Internship',
             fontsize=16, fontweight='bold')
axes = np.array(axes).flatten()

for idx, (feat, color) in enumerate(zip(feature_cols, colors)):
    axes[idx].scatter(df_num[feat], df_num[target_col],
                      alpha=0.5, color=color, s=30)
    m, b = np.polyfit(df_num[feat], df_num[target_col], 1)
    x_line = np.linspace(df_num[feat].min(), df_num[feat].max(), 100)
    axes[idx].plot(x_line, m*x_line + b, 'k--', linewidth=1.5)
    axes[idx].set_xlabel(feat)
    axes[idx].set_ylabel(target_col)
    axes[idx].set_title(f'{feat} vs {target_col}')

# Sales distribution
ax_dist = axes[n_features]
ax_dist.hist(df_num[target_col], bins=20, color='#9b59b6', edgecolor='white')
ax_dist.set_title(f'{target_col} Distribution')
ax_dist.set_xlabel(target_col)

# Correlation heatmap
if n_features + 1 < len(axes):
    ax_corr = axes[n_features + 1]
    sns.heatmap(df_num.corr(), annot=True, fmt='.2f',
                cmap='coolwarm', ax=ax_corr, square=True)
    ax_corr.set_title('Correlation Heatmap')

# Hide unused axes
for i in range(n_features + 2, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('sales_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n📈 EDA saved: sales_eda.png")

# ─────────────────────────────────────────────────────────────
# STEP 4: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
X = df_num[feature_cols]
y = df_num[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\n📦 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────
# STEP 5: TRAIN MODELS
# ─────────────────────────────────────────────────────────────
print("\n🤖 Training models...")

models = {
    'Linear Regression':    LinearRegression(),
    'Random Forest':        RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':    GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds,
                     'mae': mae, 'rmse': rmse, 'r2': r2}
    print(f"\n   {name}:")
    print(f"     MAE  : {mae:.4f}")
    print(f"     RMSE : {rmse:.4f}")
    print(f"     R²   : {r2:.4f}")

best_name = max(results, key=lambda k: results[k]['r2'])
best = results[best_name]
print(f"\n🏆 Best: {best_name} (R² = {best['r2']:.4f})")

# ─────────────────────────────────────────────────────────────
# STEP 6: EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle(f'Sales Prediction - {best_name}', fontsize=14, fontweight='bold')

# Actual vs Predicted
axes2[0].scatter(y_test, best['preds'], alpha=0.6, color='#3498db', s=40)
mn = min(y_test.min(), best['preds'].min())
mx = max(y_test.max(), best['preds'].max())
axes2[0].plot([mn, mx], [mn, mx], 'r--', linewidth=2)
axes2[0].set_xlabel(f'Actual {target_col}')
axes2[0].set_ylabel(f'Predicted {target_col}')
axes2[0].set_title(f'Actual vs Predicted\nR² = {best["r2"]:.4f}')

# Residuals
residuals = y_test.values - best['preds']
axes2[1].scatter(best['preds'], residuals, alpha=0.6, color='#e74c3c', s=40)
axes2[1].axhline(0, color='black', linestyle='--', linewidth=1.5)
axes2[1].set_xlabel('Predicted')
axes2[1].set_ylabel('Residuals')
axes2[1].set_title('Residual Plot')

# R² Comparison
names = list(results.keys())
r2s = [results[n]['r2'] for n in names]
bar_colors = ['#2ecc71' if n == best_name else '#3498db' for n in names]
bars = axes2[2].bar(names, r2s, color=bar_colors, width=0.5)
axes2[2].set_ylim(0, 1.1)
axes2[2].set_ylabel('R² Score')
axes2[2].set_title('Model Comparison (R²)')
for bar, r2 in zip(bars, r2s):
    axes2[2].text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.01, f'{r2:.3f}',
                  ha='center', fontweight='bold')
axes2[2].tick_params(axis='x', rotation=10)

plt.tight_layout()
plt.savefig('sales_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n📈 Results saved: sales_results.png")

# ─────────────────────────────────────────────────────────────
# STEP 7: SAMPLE PREDICTIONS
# ─────────────────────────────────────────────────────────────
print("\n🔮 Sample Predictions:")
print("-" * 50)
sample_input = X_test.head(3)
sample_preds = best['model'].predict(sample_input)
actual_vals  = y_test.head(3).values

for i in range(len(sample_input)):
    print(f"  Sample {i+1}: Predicted={sample_preds[i]:.2f} | Actual={actual_vals[i]:.2f}")

print("\n" + "="*60)
print("   TASK 4 COMPLETE ✅")
print("="*60)
