# W2D4 Test Results

## Execution Status

The W2D4 Python implementation was executed successfully without errors.

## Dataset

- Dataset: Breast Cancer Wisconsin
- Samples: 569
- Features: 30
- Training samples: 455
- Testing samples: 114

## Test Accuracy

| Scaling Method | Accuracy |
| -------------- | -------: |
| StandardScaler |   0.9825 |
| MinMaxScaler   |   0.9561 |
| RobustScaler   |   0.9825 |

## 5-Fold Cross-Validation

| Scaling Method | Mean Accuracy | Standard Deviation |
| -------------- | ------------: | -----------------: |
| StandardScaler |        0.9737 |             0.0166 |
| MinMaxScaler   |        0.9649 |             0.0200 |
| RobustScaler   |        0.9789 |             0.0131 |

## Final Model

- Logistic Regression
- StandardScaler
- Final Test Accuracy: 0.9825

## Status

All tests completed successfully.
