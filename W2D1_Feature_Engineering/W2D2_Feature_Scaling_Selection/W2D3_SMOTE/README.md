# W2D3 - Handling Imbalanced Data Using SMOTE

## Objective

The objective of this project is to handle imbalanced classification data using SMOTE (Synthetic Minority Over-sampling Technique).

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib

## Project Workflow

1. Created an imbalanced classification dataset.
2. Checked the class distribution.
3. Split the dataset into training and testing data.
4. Applied SMOTE only to the training data.
5. Balanced the minority class using synthetic samples.
6. Trained a Random Forest classifier.
7. Generated predictions.
8. Evaluated the model using accuracy, confusion matrix and classification report.
9. Performed tests to verify the implementation.

## Why SMOTE?

SMOTE generates synthetic samples for the minority class instead of simply duplicating existing samples. This helps the machine learning model learn from both classes more effectively.

## Important Design Decision

SMOTE was applied only to the training data after train-test splitting. This helps prevent data leakage from the test set.

## Testing

The project includes tests to verify:

- SMOTE balances the training classes.
- Prediction count matches the test data.
- Model accuracy is within a valid range.

## Result

The training dataset was successfully balanced using SMOTE, and the Random Forest model was trained and evaluated on the original test data.
