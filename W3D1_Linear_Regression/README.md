# W3D1 - Linear Regression

## Objective

Implement and evaluate Linear Regression using Scikit-Learn and compare it with Ridge and Lasso Regression.

## Dataset

The project uses the Diabetes dataset available in Scikit-Learn.

## Tasks Completed

- Loaded and explored the Diabetes dataset
- Split data into training and testing sets
- Trained Linear Regression model
- Printed model intercept and coefficients
- Generated predictions
- Evaluated the model using:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score
- Created Actual vs Predicted visualization
- Created Residual plot
- Trained Ridge Regression
- Trained Lasso Regression
- Compared Linear, Ridge, and Lasso models
- Saved model comparison results as CSV

## Models

### Linear Regression

Ordinary Least Squares Linear Regression was used as the baseline model.

### Ridge Regression

Ridge Regression uses L2 regularization to reduce the effect of large coefficients and help control overfitting.

### Lasso Regression

Lasso Regression uses L1 regularization and can reduce some coefficients to zero, which can help with feature selection.

## Evaluation Metrics

The models were compared using:

- MSE
- RMSE
- MAE
- R² Score

## Project Structure

```text
W3D1_Linear_Regression/
│
├── W3D1_Linear_Regression.ipynb
├── README.md
└── output/
    └── linear_regression_results.csv
```
