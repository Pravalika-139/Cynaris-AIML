# W3D3 - Decision Trees & Random Forests

## Objective

Implement and compare Decision Tree and Random Forest classification models using Python and Scikit-learn.

## Dataset

The Breast Cancer dataset provided by Scikit-learn was used.

- Samples: 569
- Features: 30
- Problem Type: Binary Classification

## Tasks Completed

- Loaded and inspected the dataset
- Checked missing values
- Performed train-test split
- Built a Decision Tree classifier
- Evaluated Decision Tree performance
- Visualized the Decision Tree
- Checked for overfitting
- Tuned Decision Tree hyperparameters
- Built a Random Forest classifier
- Compared model performance
- Generated a confusion matrix
- Analyzed feature importance
- Generated a classification report

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Machine Learning Concepts

### Decision Tree

A Decision Tree is a supervised learning algorithm that makes predictions by splitting data into branches based on feature values.

### Gini Impurity

Gini impurity measures how mixed the classes are in a node. A lower Gini impurity represents a purer node.

### Information Gain

Information gain measures how much a split improves the separation of classes.

### Overfitting

A Decision Tree can overfit when it becomes too deep and learns noise from the training data. Hyperparameters such as `max_depth`, `min_samples_split`, and `min_samples_leaf` can help control overfitting.

### Random Forest

Random Forest combines multiple Decision Trees and uses their combined predictions to produce a more robust classification result.

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

## Conclusion

Decision Trees provide an interpretable model that can be visualized easily. However, a single Decision Tree can overfit the training data.

Hyperparameter tuning was used to control the complexity of the Decision Tree.

Random Forest combines multiple Decision Trees and generally provides a more robust and stable classification model.

Therefore, comparing both models helps identify the model that provides better generalization on unseen data.

## Future Improvements

- Use GridSearchCV for systematic hyperparameter tuning
- Apply cross-validation for more reliable model evaluation
- Compare additional classification algorithms
- Track experiments using MLflow
- Add automated tests for model performance
