# -----------------------------
# Import Libraries
# -----------------------------

import numpy as np
import pandas as pd
import pickle

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# -----------------------------
# Load Dataset
# -----------------------------

columns = [
    "pregnancies",
    "glucose",
    "bloodpressure",
    "skinthickness",
    "insulin",
    "bmi",
    "diabetespedigreefunction",
    "age",
    "outcome"
]

df = pd.read_csv("data/pima-indians-diabetes.csv", names=columns)

# -----------------------------
# Data Preprocessing
# -----------------------------

cols = ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi']

df[cols] = df[cols].replace(0, np.nan)

imputer = SimpleImputer(strategy="median")
df[cols] = imputer.fit_transform(df[cols])

# -----------------------------
# Features and Target
# -----------------------------

X = df.drop("outcome", axis=1)
y = df["outcome"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Evaluation Function
# -----------------------------

results = {}

def evaluate_model(model, X_test_data, name):

    preds = model.predict(X_test_data)
    probs = model.predict_proba(X_test_data)[:,1]

    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    results[name] = acc

    print(f"\n----- {name} -----")
    print("Accuracy:", round(acc,4))
    print("Precision:", round(precision,4))
    print("Recall:", round(recall,4))
    print("F1 Score:", round(f1,4))
    print("ROC AUC:", round(roc,4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# -----------------------------
# Train Models
# -----------------------------

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
evaluate_model(lr, X_test_scaled, "Logistic Regression")

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
evaluate_model(rf, X_test, "Random Forest")

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
evaluate_model(gb, X_test, "Gradient Boosting")

# Support Vector Machine
svc = SVC(probability=True)
svc.fit(X_train_scaled, y_train)
evaluate_model(svc, X_test_scaled, "Support Vector Machine")

# XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)
evaluate_model(xgb, X_test, "XGBoost")

# -----------------------------
# Select Best Model
# -----------------------------

best_model_name = max(results, key=results.get)

print("\nBest Model:", best_model_name)

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "Support Vector Machine": svc,
    "XGBoost": xgb
}

best_model = models[best_model_name]

# -----------------------------
# Save Model
# -----------------------------

pickle.dump(best_model, open("model/diabetes_model.pk1", "wb"))
pickle.dump(scaler, open("model/scaler.pk1", "wb"))

print("\nModel saved successfully.")