# PythonExercise
This code demonstrates the performance comparison between two machine learning algorithms: Logistic Regression and Decision Tree, using the breast cancer dataset. It employs grid search to optimize the hyperparameters of each model and evaluates their performance based on accuracy, precision, recall, f1-score, and confusion matrix. The results indicate that Logistic Regression with penalty = l2 and C = 10 outperforms the Decision Tree with criterion = gini and min_samples_split = 3, achieving a higher accuracy of 96.49%. This experiment highlights the importance of hyperparameter tuning and the potential of Logistic Regression for accurate breast cancer classification.

Initialization:

The code starts by importing the necessary libraries: numpy, sklearn.datasets, sklearn.tree, sklearn.linear_model, sklearn.model_selection, sklearn.preprocessing, sklearn.metrics.

These libraries are used for data loading, preprocessing, model training, evaluation, and metrics calculation.

Data loading:

The code loads the breast cancer data from the sklearn.datasets library.

The data is split into two parts: training data (80%) and testing data (20%).

The training data is further preprocessed using MinMaxScaler to normalize the features.

Logistic Regression Model:

The code defines a LogisticRegression model.

The model is then tuned using GridSearchCV to find the best combination of hyperparameters.

The hyperparameters tuned are penalty (l1 or l2) and C (regularization parameter).

The best hyperparameters are found to be penalty = l2 and C = 10.

The trained model is evaluated using accuracy, precision, recall, f1-score, and confusion matrix.

Decision Tree Model:

The code defines a DecisionTreeClassifier model.

The model is then tuned using GridSearchCV to find the best combination of hyperparameters.

The hyperparameters tuned are criterion (entropy or gini) and min_samples_split (the minimum number of samples required to split a node).

The best hyperparameters are found to be criterion = gini and min_samples_split = 3.

The trained model is evaluated using accuracy, precision, recall, f1-score, and confusion matrix.

Overall:

The code performs a grid search on two different models - Logistic Regression and Decision Tree - to find the best hyperparameters for each model.

The best models are then evaluated and their performance is compared.

The code shows that the Logistic Regression model with the best hyperparameters (penalty = l2 and C = 10) achieves a higher accuracy (96.49%) than the Decision Tree model with the best hyperparameters (criterion = gini and min_samples_split = 3) which achieves an accuracy of 92.39%.
