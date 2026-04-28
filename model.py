import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Step 1: Load & prepare ─────────────────────────────
print("Loading data...")
df = pd.read_csv('data/ai4i2020.csv')

# Encode machine type
le = LabelEncoder()
df['Type_encoded'] = le.fit_transform(df['Type'])

# Features and target
features = ['Type_encoded', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = df[features]
y = df['Machine failure']

print(f"Total samples  : {len(df)}")
print(f"Failures       : {y.sum()} ({y.mean()*100:.2f}%)")
print(f"Healthy        : {(y==0).sum()} ({(y==0).mean()*100:.2f}%)")

# ── Step 2: Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining rows  : {len(X_train)}")
print(f"Testing rows   : {len(X_test)}")

# ── Step 3: Scale ──────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Step 4: SMOTE ─────────────────────────────────────
print("\nApplying SMOTE...")
sm = SMOTE(random_state=42)
X_train_r, y_train_r = sm.fit_resample(X_train_s, y_train)
print(f"After SMOTE    : {len(X_train_r)} rows")

# ── Step 5: Helper ────────────────────────────────────
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cm    = confusion_matrix(y_test, preds)
    auc   = roc_auc_score(y_test, proba)
    prec  = precision_score(y_test, preds, zero_division=0)
    rec   = recall_score(y_test, preds, zero_division=0)
    f1    = f1_score(y_test, preds, zero_division=0)
    print(f"\n── {name} ──")
    print(f"AUC-ROC   : {auc:.3f}")
    print(f"Precision : {prec:.3f}")
    print(f"Recall    : {rec:.3f}")
    print(f"F1 Score  : {f1:.3f}")
    print(f"Caught    : {cm[1][1]} | Missed: {cm[1][0]} | False alarms: {cm[0][1]}")
    return {
        'Model': name, 'AUC-ROC': round(auc,3),
        'Precision': round(prec,3), 'Recall': round(rec,3),
        'F1 Score': round(f1,3),
        'Caught': int(cm[1][1]), 'Missed': int(cm[1][0]),
        'False Alarms': int(cm[0][1])
    }

results = []

# ── Step 6: Train all 6 models ────────────────────────
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_r, y_train_r)
results.append(evaluate("Logistic Regression", lr, X_test_s, y_test))

print("\nTraining Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_r, y_train_r)
results.append(evaluate("Decision Tree", dt, X_test_s, y_test))

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_r, y_train_r)
results.append(evaluate("Random Forest", rf, X_test_s, y_test))

print("\nTraining Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_r, y_train_r)
results.append(evaluate("Gradient Boosting", gb, X_test_s, y_test))

print("\nTraining XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42,
                    eval_metric='logloss', verbosity=0)
xgb.fit(X_train_r, y_train_r)
results.append(evaluate("XGBoost", xgb, X_test_s, y_test))

print("\nTraining KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_r, y_train_r)
results.append(evaluate("KNN", knn, X_test_s, y_test))

# ── Step 7: Comparison table ──────────────────────────
results_df = pd.DataFrame(results).sort_values('Recall', ascending=False)
print("\n\n── Final Comparison (sorted by Recall) ──")
print(f"{'Model':<22} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Caught':>7} {'Missed':>7} {'FalseAlarm':>11}")
print("-" * 80)
for r in results_df.itertuples():
    print(f"{r.Model:<22} {r._2:>6} {r.Precision:>6} {r.Recall:>6} {r._5:>6} {r.Caught:>7} {r.Missed:>7} {r._8:>11}")

# ── Step 8: Save best model ───────────────────────────
best_name = results_df.iloc[0]['Model']
models_dict = {
    'Logistic Regression': lr,
    'Decision Tree': dt,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'XGBoost': xgb,
    'KNN': knn
}
joblib.dump(models_dict[best_name], 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features, 'feature_names.pkl')
results_df.to_csv('model_results.csv', index=False)
print(f"\nBest model by Recall: {best_name} — saved!")

# ── Step 9: Failure type breakdown ────────────────────
print("\n── Failure Type Breakdown ──")
failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
for ft in failure_types:
    count = df[ft].sum()
    print(f"{ft}: {count} failures ({count/len(df)*100:.2f}%)")