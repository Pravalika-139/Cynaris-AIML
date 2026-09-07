# W2D4: Train/Test Split & Cross-Validation

## Objective

The objective of this task is to understand and implement:

- Train/Test Split
- Feature Scaling
- StandardScaler
- MinMaxScaler
- RobustScaler
- Cross-Validation
- Stratified K-Fold Cross-Validation

## Dataset

The Breast Cancer Wisconsin dataset from Scikit-learn was used.

- Samples: 569
- Features: 30
- Problem type: Binary Classification

## Train/Test Split

The dataset was divided into:

- 80% training data
- 20% testing data

`random_state=42` was used to make the results reproducible.

Stratification was used to maintain the class distribution in the training
and testing datasets.

## Feature Scaling

Three scaling methods were compared.

### StandardScaler

StandardScaler transforms features so that they have a mean close to 0
and a standard deviation close to 1.

### MinMaxScaler

MinMaxScaler transforms features to a fixed range, usually between 0 and 1.

### RobustScaler

RobustScaler uses the median and interquartile range, making it more
resistant to outliers.

## Machine Learning Model

Logistic Regression was used to compare the performance of the three
scaling methods.

A Scikit-learn Pipeline was used to combine feature scaling and the model.

## Cross-Validation

5-Fold Stratified Cross-Validation was used.

The dataset is divided into five folds. Each fold is used once for validation
while the remaining four folds are used for training.

## Data Leakage Prevention

A Pipeline was used so that the scaler is fitted only on the training
portion of the data during each cross-validation fold.

This helps prevent data leakage.

## Results

### Test Set Accuracy

Enter the actual results obtained from the notebook:

| Scaling Method | Test Accuracy |
| -------------- | ------------: |
| StandardScaler |    Add result |
| MinMaxScaler   |    Add result |
| RobustScaler   |    Add result |

### Cross-Validation Accuracy

Enter the actual mean accuracy obtained from the notebook:

| Scaling Method | Mean CV Accuracy |
| -------------- | ---------------: |
| StandardScaler |       Add result |
| MinMaxScaler   |       Add result |
| RobustScaler   |       Add result |

## Conclusion

Train/Test Split and Cross-Validation were successfully implemented.

StandardScaler, MinMaxScaler, and RobustScaler were compared using
Logistic Regression.

The experiment shows why feature scaling is important when numerical
features have different ranges.

Cross-Validation provides a more reliable estimate of model performance
than relying on a single train/test split.

## Files

- `W2D4_Train_Test_CrossValidation.ipynb` - Notebook containing the complete analysis.
- `train_test_scaling.py` - Python implementation.
- `requirements.txt` - Required Python packages.

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Results

### Test Set Accuracy

| Scaling Method | Test Accuracy |
| -------------- | ------------: |
| StandardScaler |        0.9825 |
| MinMaxScaler   |        0.9737 |
| RobustScaler   |        0.9825 |

### 5-Fold Cross-Validation

| Scaling Method | Mean CV Accuracy | Standard Deviation |
| -------------- | ---------------: | -----------------: |
| StandardScaler |           0.9737 |             0.0166 |
| MinMaxScaler   |           0.9649 |             0.0200 |
| RobustScaler   |           0.9789 |             0.0131 |

### Final Model

- Model: Logistic Regression
- Scaler: StandardScaler
- Final Test Accuracy: **0.9825**

## Analysis

RobustScaler achieved the highest mean cross-validation accuracy of 0.9789
and the lowest standard deviation of 0.0131 among the three scaling methods.

StandardScaler achieved a final test accuracy of 0.9825.

MinMaxScaler produced the lowest mean cross-validation accuracy of 0.9649
in this experiment.

The results show that the choice of scaling method can affect model
performance.

## Results

### Test Set Accuracy

| Scaling Method | Test Accuracy |
| -------------- | ------------: |
| StandardScaler |        0.9825 |
| MinMaxScaler   |        0.9561 |
| RobustScaler   |        0.9825 |

### 5-Fold Cross-Validation

| Scaling Method | Mean CV Accuracy | Standard Deviation |
| -------------- | ---------------: | -----------------: |
| StandardScaler |           0.9737 |             0.0166 |
| MinMaxScaler   |           0.9649 |             0.0200 |
| RobustScaler   |           0.9789 |             0.0131 |

### Final Model

- Model: Logistic Regression
- Scaler: StandardScaler
- Final Test Accuracy: **0.9825**

## Analysis

RobustScaler achieved the highest mean cross-validation accuracy of 0.9789
and the lowest standard deviation of 0.0131 among the three scaling methods.

StandardScaler achieved a final test accuracy of 0.9825.

MinMaxScaler produced the lowest test accuracy of 0.9561 and the lowest
mean cross-validation accuracy of 0.9649 in this experiment.

The results demonstrate that the choice of scaling method can affect
machine learning model performance.
