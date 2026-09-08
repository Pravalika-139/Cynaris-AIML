# W4D1: Model Evaluation Metrics

## Objective

Evaluate a classification model using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

The project also demonstrates K-Fold Cross-Validation, Stratified K-Fold Cross-Validation, Learning Curves, and MLflow experiment tracking.

## Tools and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- MLflow

## Dataset

A synthetic binary classification dataset was generated using Scikit-learn's `make_classification`.

- Samples: 1000
- Features: 10
- Classes: 2

## Model

Logistic Regression was used as the classification algorithm.

A Scikit-learn Pipeline was created with:

1. StandardScaler
2. LogisticRegression

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report

## Cross-Validation

Two cross-validation approaches were implemented:

### K-Fold Cross-Validation

The dataset was divided into 5 folds and evaluated using accuracy.

### Stratified K-Fold Cross-Validation

Stratified K-Fold was used to preserve class proportions across the folds.

## Learning Curve

Learning curves were generated to compare training and validation performance.

They help identify:

- Underfitting
- Overfitting
- Good model fit

## MLflow

MLflow was used to track:

- Model type
- Cross-validation method
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross-validation performance

## Conclusion

Multiple evaluation metrics provide a better understanding of classification performance than accuracy alone.

Cross-validation helps estimate model generalization, while learning curves help diagnose overfitting and underfitting.

MLflow provides experiment tracking for reproducible model evaluation.
