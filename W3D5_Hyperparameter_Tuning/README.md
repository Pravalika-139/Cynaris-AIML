# W3D5: Hyperparameter Tuning — GridSearch & RandomSearch

## Objective

The objective of this project is to understand and implement
hyperparameter tuning using GridSearchCV and RandomizedSearchCV.

Support Vector Machine (SVM) and K-Nearest Neighbours (KNN)
classification algorithms are trained, tuned, and compared.

## Dataset

The Breast Cancer Wisconsin dataset available through
scikit-learn was used.

- Samples: 569
- Features: 30
- Classes: 2
- Classes: Malignant and Benign

## Algorithms Used

### 1. Support Vector Machine (SVM)

SVM is a supervised machine learning algorithm that finds an
optimal decision boundary between classes.

### 2. K-Nearest Neighbours (KNN)

KNN classifies a data point based on the classes of its nearest
neighbours.

## Hyperparameter Tuning

### GridSearchCV

GridSearchCV performs an exhaustive search over all specified
hyperparameter combinations.

For SVM, parameters such as:

- C
- kernel
- gamma

were tuned.

For KNN, parameters such as:

- n_neighbors
- weights
- metric

were tuned.

### RandomizedSearchCV

RandomizedSearchCV tests a fixed number of randomly selected
parameter combinations.

It can be faster than exhaustive grid search when the
hyperparameter search space is large.

## Preprocessing

StandardScaler was used to standardize the features.

A scikit-learn Pipeline was used so that scaling and model
training are handled together during cross-validation.

This helps prevent data leakage during model evaluation.

## Model Evaluation

The following evaluation methods were used:

- 5-fold cross-validation
- Test accuracy
- Classification report
- Confusion matrix

Both baseline and tuned models were compared.

## Key Observations

Hyperparameter tuning helps identify better parameter settings
for machine learning algorithms.

GridSearchCV provides a systematic exhaustive search, while
RandomizedSearchCV explores a selected number of combinations.

The final model was selected based on test-set performance after
comparing the tuned SVM and KNN models.

## Conclusion

This project demonstrates how hyperparameter optimization can
improve model selection and performance.

SVM and KNN were successfully tuned using GridSearchCV and
RandomizedSearchCV and evaluated using cross-validation and
test-set metrics.
