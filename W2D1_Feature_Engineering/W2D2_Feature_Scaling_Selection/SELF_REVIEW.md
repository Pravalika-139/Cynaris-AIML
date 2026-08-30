# W2D2 Feature Scaling & Selection — Self Review

## Practical Tasks

- [x] Applied LabelEncoder on categorical data
- [x] Applied OneHotEncoder on categorical data
- [x] Applied OrdinalEncoder on categorical data
- [x] Compared encoding techniques and documented trade-offs
- [x] Applied StandardScaler
- [x] Applied MinMaxScaler
- [x] Applied RobustScaler
- [x] Compared scaling methods
- [x] Plotted distributions before/after scaling
- [x] Applied SelectKBest
- [x] Identified top 5 features
- [x] Documented why selected features matter
- [x] Answered viva questions
- [x] Added final conclusion

## Scaling Summary

| Method         | Purpose                         | Outlier Sensitivity |
| -------------- | ------------------------------- | ------------------- |
| StandardScaler | Standardizes data around mean 0 | Sensitive           |
| MinMaxScaler   | Scales values between 0 and 1   | Highly sensitive    |
| RobustScaler   | Uses median and IQR             | Less sensitive      |

## Feature Selection

SelectKBest with ANOVA F-test was used to identify the top 5 numerical features related to the target variable `Grade`.

## Viva Preparation

- OneHotEncoder is suitable for unordered categories.
- OrdinalEncoder is suitable for ordered categories.
- StandardScaler is affected by outliers because it uses mean and standard deviation.
- Feature leakage is prevented by fitting preprocessing steps only on training data.

## Status

W2D2 Feature Scaling & Selection practical work completed.

## Git Progress

- Commit 1 completed: Feature scaling and selection implementation.
- Notebook and self-review documentation committed successfully.
- Ready for final Git push and Pull Request.
