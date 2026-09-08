# W4D2: Bias-Variance Tradeoff & Regularisation

## Objective

Understand the bias-variance tradeoff and compare L1 and L2 regularisation. Use systematic hyperparameter search techniques such as GridSearchCV and RandomizedSearchCV.

## Dataset

The Diabetes dataset from Scikit-learn was used for the regression experiments.

- Samples: 442
- Features: 10
- Target: Diabetes disease progression measure

## Tasks Completed

1. Loaded and explored the Diabetes dataset.
2. Split the data into training and testing sets.
3. Applied StandardScaler using Scikit-learn pipelines.
4. Implemented Ridge Regression (L2 regularisation).
5. Implemented Lasso Regression (L1 regularisation).
6. Compared different Ridge regularisation strengths.
7. Used GridSearchCV to systematically find the best Ridge alpha.
8. Used RandomizedSearchCV to search a larger hyperparameter space.
9. Compared Grid Search and Randomized Search performance.
10. Visualised the effect of regularisation strength on training and testing RMSE.
11. Examined Lasso coefficients for feature selection.

## Key Concepts

### Bias-Variance Tradeoff

High bias means the model is too simple and may underfit the data.

High variance means the model is too sensitive to training data and may overfit.

Regularisation helps control model complexity and can improve generalisation.

### Ridge Regression (L2)

Ridge adds an L2 penalty to the loss function. It shrinks coefficients toward zero while generally keeping all features in the model.

### Lasso Regression (L1)

Lasso adds an L1 penalty. It can shrink some coefficients exactly to zero, making it useful for feature selection.

### GridSearchCV

Grid Search evaluates every specified combination of hyperparameters. It is useful when the search space is relatively small and well defined.

### RandomizedSearchCV

Randomized Search evaluates a fixed number of randomly selected hyperparameter combinations. It is useful when the search space is larger.

## Result

The experiments demonstrated how regularisation strength affects model performance and how systematic hyperparameter search can be used instead of manually guessing parameter values.

## Tools Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Git/GitHub

## Conclusion

Grid Search and Randomized Search provide systematic approaches for hyperparameter tuning. Ridge and Lasso regularisation help control model complexity and improve generalisation by reducing the effects of overfitting.

## Self-Review Checklist

- [x] Code is clean and commented.
- [x] Dataset was loaded and split correctly.
- [x] Ridge and Lasso regularisation were implemented.
- [x] GridSearchCV was implemented.
- [x] RandomizedSearchCV was implemented.
- [x] Model performance was evaluated.
- [x] Regularisation results were visualised.
- [x] Notebook outputs were generated.
- [x] README documentation was completed.
- [ ] Git changes committed with at least 2 descriptive commits.
- [ ] Changes pushed to the feature branch.
- [ ] Pull Request raised.
