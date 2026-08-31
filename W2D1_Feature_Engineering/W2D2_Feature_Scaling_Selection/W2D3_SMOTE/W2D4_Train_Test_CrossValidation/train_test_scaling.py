# W2D4: Train/Test Split, Cross-Validation & Feature Scaling

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test Split")
print("----------------")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Define scalers
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}


# Compare scalers using test data
print("\nScaling Results")
print("===============")

for scaler_name, scaler in scalers.items():

    pipeline = Pipeline([
        ("scaler", scaler),
        ("model", LogisticRegression(max_iter=5000))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"{scaler_name}: {accuracy:.4f}")


# 5-Fold Stratified Cross-Validation
print("\nCross-Validation Results")
print("========================")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for scaler_name, scaler in scalers.items():

    pipeline = Pipeline([
        ("scaler", scaler),
        ("model", LogisticRegression(max_iter=5000))
    ])

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print(f"\n{scaler_name}")
    print("Fold scores:", np.round(scores, 4))
    print("Mean accuracy:", round(scores.mean(), 4))
    print("Standard deviation:", round(scores.std(), 4))


# Final model
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)

print("\nFinal Model Performance")
print("=======================")
print(
    "Accuracy:",
    round(accuracy_score(y_test, final_predictions), 4)
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        final_predictions,
        target_names=data.target_names
    )
)