# W2D5: Titanic End-to-End Preprocessing Pipeline

## Objective

Build an end-to-end preprocessing pipeline for the Titanic dataset.

## Steps Performed

1. Loaded the Titanic dataset using Seaborn.
2. Performed Exploratory Data Analysis (EDA).
3. Checked and handled missing values.
4. Split the data into training and testing sets.
5. Encoded categorical features using One-Hot Encoding.
6. Scaled numerical features using StandardScaler.
7. Used Scikit-learn Pipeline and ColumnTransformer.
8. Verified that the processed data contains no missing values.
9. Exported ML-ready training and testing datasets.

## Input Features

- pclass
- sex
- age
- sibsp
- parch
- fare
- embarked

## Target

- survived

## Output Files

- `output/titanic_train_ml_ready.csv`
- `output/titanic_test_ml_ready.csv`

## Results

- Training samples: 712
- Testing samples: 179
- Final processed columns: 11
- Missing values after preprocessing: 0
