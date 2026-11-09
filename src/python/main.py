# Q13 - Load Titanic training data

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

data_path = Path(__file__).resolve().parent.parent / "data"

train = pd.read_csv(data_path / "train.csv")

print("Training data shape:", train.shape)
print(train.head())

# Q14 - Explore and clean data

print("\nMissing values:")
print(train.isnull().sum())

print("\nSurvival rate:")
print(train["Survived"].mean())

print("\nSurvival rate by sex:")
print(train.groupby("Sex")["Survived"].mean())

# Select features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

train_clean = train[features + ["Survived"]].copy()

# Fill missing values
train_clean["Age"] = train_clean["Age"].fillna(train_clean["Age"].median())
train_clean["Fare"] = train_clean["Fare"].fillna(train_clean["Fare"].median())
train_clean["Embarked"] = train_clean["Embarked"].fillna(train_clean["Embarked"].mode()[0])

# Convert text to numeric columns
train_clean = pd.get_dummies(
    train_clean,
    columns=["Sex", "Embarked"],
    dtype=int
)

print("\nCleaned data:")
print(train_clean.head())

print("\nMissing values after cleaning:")
print(train_clean.isnull().sum())

# Q15 - Logistic Regression

X = train_clean.drop(columns=["Survived"])
y = train_clean["Survived"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("\nLogistic Regression model trained successfully!")
print("Features used:", X.columns.tolist())


# Q16 - Training Accuracy

from sklearn.metrics import accuracy_score
train_pred = model.predict(X)
train_accuracy = accuracy_score(y, train_pred)
print("\nTraining Accuracy:", round(train_accuracy, 4))

# Q17 - Predict on test data

test = pd.read_csv(data_path / "test.csv")

test_clean = test[features].copy()

test_clean["Age"] = test_clean["Age"].fillna(train["Age"].median())
test_clean["Fare"] = test_clean["Fare"].fillna(train["Fare"].median())
test_clean["Embarked"] = test_clean["Embarked"].fillna(train["Embarked"].mode()[0])

test_clean = pd.get_dummies(
    test_clean,
    columns=["Sex", "Embarked"],
    dtype=int
)

test_clean = test_clean.reindex(columns=X.columns, fill_value=0)

test_pred = model.predict(test_clean)

print("\nTest data shape:", test.shape)
print("Test predictions:", test_pred[:10])
print("Total predictions:", len(test_pred))


# Q18 - Test accuracy using a holdout set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

test_model = LogisticRegression(max_iter=1000)
test_model.fit(X_train, y_train)

y_pred = test_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

print("\nHoldout Test Accuracy:", round(test_accuracy, 4))
print("Holdout test samples:", len(y_test))
print("Note: Kaggle test.csv has no true survival labels.")

# Q19 - Summary of results

print("\n===== Titanic Analysis Summary =====")
print("Training data:", train.shape)
print("Cleaned data:", train_clean.shape)
print("Training accuracy:", round(train_accuracy, 4))
print("Kaggle test predictions:", len(test_pred))
print("Holdout test accuracy:", round(test_accuracy, 4))
print("====================================")
