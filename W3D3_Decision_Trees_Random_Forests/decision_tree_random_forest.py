# W3D3 - Decision Trees & Random Forests

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(
    data.target,
    name="target"
)

print("Dataset loaded successfully!")
print("Features:", X.shape)
print("Target:", y.shape)


# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Decision Tree
# --------------------------------------------------

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)


# --------------------------------------------------
# 4. Tuned Decision Tree
# --------------------------------------------------

tuned_dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

tuned_dt.fit(X_train, y_train)

y_pred_tuned = tuned_dt.predict(X_test)


# --------------------------------------------------
# 5. Random Forest
# --------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)


# --------------------------------------------------
# 6. Model Evaluation
# --------------------------------------------------

models = {
    "Decision Tree": y_pred_dt,
    "Tuned Decision Tree": y_pred_tuned,
    "Random Forest": y_pred_rf
}

for model_name, predictions in models.items():

    print("\n" + "=" * 40)
    print(model_name)
    print("=" * 40)

    print(
        f"Accuracy : {accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Precision: {precision_score(y_test, predictions):.4f}"
    )

    print(
        f"Recall   : {recall_score(y_test, predictions):.4f}"
    )

    print(
        f"F1 Score : {f1_score(y_test, predictions):.4f}"
    )


# --------------------------------------------------
# 7. Classification Report
# --------------------------------------------------

print("\nRandom Forest Classification Report")
print("------------------------------------")

print(
    classification_report(
        y_test,
        y_pred_rf,
        target_names=data.target_names
    )
)


# --------------------------------------------------
# 8. Confusion Matrix
# --------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf,
    display_labels=data.target_names
)

plt.title("Random Forest Confusion Matrix")
plt.show()


# --------------------------------------------------
# 9. Feature Importance
# --------------------------------------------------

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=data.feature_names
).sort_values(ascending=False)

print("\nTop 10 Important Features")
print("--------------------------")

print(feature_importance.head(10))


# --------------------------------------------------
# 10. Feature Importance Visualization
# --------------------------------------------------

feature_importance.head(10).sort_values().plot(
    kind="barh",
    figsize=(10, 6)
)

plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


# --------------------------------------------------
# 11. Decision Tree Visualization
# --------------------------------------------------

plt.figure(figsize=(20, 10))

plot_tree(
    dt_model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    max_depth=3
)

plt.title("Decision Tree Visualization")
plt.show()