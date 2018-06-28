#! /usr/bin/python3
# Creating a model to predict survival of Titanic passengers

# Step 3 - EDA
import pandas as pd

titanic_training = pd.read_csv('../datasets/train.csv')
titanic_test = pd.read_csv('../datasets/test.csv')

print('titanic_training.shape:')
print(titanic_training.shape)
print('titanic_test.shape:')
print(titanic_test.shape)

print('titanic_training.describe():')
print(titanic_training.describe())

data = titanic_training[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].as_matrix()
import numpy as np

age = data[:, 2]
print('Number of NaN values in age data: ')
print(np.count_nonzero(np.isnan(age)))
age = age[np.logical_not(np.isnan(age))]

from scipy import stats

print('Numerical data, stats.describe(data): ')
print(stats.describe(data))
print('Age with NaNs removed, stats.describe(age):')
print(stats.describe(age))

import matplotlib.pyplot as plt

sibsp = data[:, 3]
fig1, ax1 = plt.subplots()
ax1.hist(sibsp, 20, histtype='bar', label='SibSp')
ax1.legend()
ax1.set(title='Distribution of SibSp', ylabel='Number of datapoints', xlabel='SibSp data')
fig1.savefig('../gitIgnoreDir/SibSp_histogram.png')

fare = data[:, 5]
fig2, ax2 = plt.subplots()
ax2.hist(fare, 100, histtype='bar', label='Fare')
ax2.legend()
ax2.set(title='Distribution of Fare', ylabel='Number of datapoints', xlabel='Fare data')
fig2.savefig('../gitIgnoreDir/Fare_histogram.png')

fig3, ax3 = plt.subplots()
ax3.hist(age, 100, histtype='bar', label='Age')
ax3.legend()
ax3.set(title='Distribution of Age', ylabel='Number of datapoints', xlabel='Age data')
fig3.savefig('../gitIgnoreDir/Age_histogram.png')

# training
# 0sub
training_data_0 = titanic_training[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].as_matrix()
train_nan_indicies_0 = np.isnan(training_data_0[:, 1])
training_data_0[:, 1][train_nan_indicies_0] = 0
# meansub
training_data_mean = titanic_training[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].as_matrix()
training_age_mean = np.nanmean(training_data_mean, axis=0)[1]
train_nan_indicies_mean = np.isnan(training_data_mean[:, 1])
training_data_mean[:, 1][train_nan_indicies_mean] = training_age_mean
# testing
# 0sub
test_data_0 = titanic_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].as_matrix()
test_nan_indicies_0 = np.isnan(test_data_0[:, 1])
test_data_0[:, 1][test_nan_indicies_0] = 0
# meansub
test_data_mean = titanic_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].as_matrix()
test_age_mean = np.nanmean(test_data_mean, axis=0)[1]
test_nan_indicies_mean = np.isnan(test_data_mean[:, 1])
test_data_mean[:, 1][test_nan_indicies_mean] = test_age_mean

target_data = titanic_training['Survived']

# Step 4 - Feature engineering
# Polynomial feature engineering
from sklearn.model_selection import train_test_split

# 0sub
X_train_0, X_validate_0, y_train_0, y_validate_0 = train_test_split(
    training_data_0,
    target_data,
    test_size=0.3)
# meansub
X_train_mean, X_validate_mean, y_train_mean, y_validate_mean = train_test_split(
    training_data_mean,
    target_data,
    test_size=0.3)

# mean normalise the training data
from sklearn.preprocessing import StandardScaler

# 0sub
scaler_0 = StandardScaler().fit(X_train_0)
standardised_X_train_0 = scaler_0.transform(X_train_0)
standardised_X_validate_0 = scaler_0.transform(X_validate_0)
# meansub
scaler_mean = StandardScaler().fit(X_train_mean)
standardised_X_train_mean = scaler_mean.transform(X_train_mean)
standardised_X_validate_mean = scaler_mean.transform(X_validate_mean)

# make some 2nd order polynomial training data
from sklearn.preprocessing import PolynomialFeatures

# 0sub
poly_2_0 = PolynomialFeatures(2)
poly_2_0.fit(X_train_0)
X_train_0_poly = poly_2_0.transform(X_train_0)
X_validate_0_poly = poly_2_0.transform(X_validate_0)
# meansub
poly_2_mean = PolynomialFeatures(2)
poly_2_mean.fit(X_train_mean)
X_train_mean_poly = poly_2_mean.transform(X_train_mean)
X_validate_mean_poly = poly_2_mean.transform(X_validate_mean)

# mean normalise that
# 0sub
scaler_0_poly = StandardScaler().fit(X_train_0_poly)
standardised_X_train_0_poly = scaler_0_poly.transform(X_train_0_poly)
standardised_X_validate_0_poly = scaler_0_poly.transform(X_validate_0_poly)
# meansub
scaler_mean_poly = StandardScaler().fit(X_train_mean_poly)
standardised_X_train_mean_poly = scaler_mean_poly.transform(X_train_mean_poly)
standardised_X_validate_mean_poly = scaler_mean_poly.transform(X_validate_mean_poly)

# Step 5 - Model creation and training
# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_original_0sub = LogisticRegression(max_iter=500, solver='lbfgs')
lr_polynomial_0sub = LogisticRegression(max_iter=500, solver='lbfgs')
lr_original_meanSub = LogisticRegression(max_iter=500, solver='lbfgs')
lr_polynomial_meanSub = LogisticRegression(max_iter=500, solver='lbfgs')

lr_original_0sub.fit(standardised_X_train_0, y_train_0)
lr_original_meanSub.fit(standardised_X_train_mean, y_train_mean)
lr_polynomial_0sub.fit(standardised_X_train_0_poly, y_train_0)
lr_polynomial_meanSub.fit(standardised_X_train_mean_poly, y_train_mean)

# Neural Network
from sklearn.neural_network import MLPClassifier

ann_original_0sub = MLPClassifier(hidden_layer_sizes=(100, 50, 25, 12), activation='logistic')
ann_polynomial_0sub = MLPClassifier(hidden_layer_sizes=(100, 50, 25, 12), activation='logistic')
ann_original_meanSub = MLPClassifier(hidden_layer_sizes=(100, 50, 25, 12), activation='logistic')
ann_polynomial_meanSub = MLPClassifier(hidden_layer_sizes=(100, 50, 25, 12), activation='logistic')

ann_original_0sub.fit(standardised_X_train_0, y_train_0)
ann_original_meanSub.fit(standardised_X_train_mean, y_train_mean)
ann_polynomial_0sub.fit(standardised_X_train_0_poly, y_train_0)
ann_polynomial_meanSub.fit(standardised_X_train_mean_poly, y_train_mean)

# Step 6 - Evaluation of model on Validation set
# Logistic Regression
pred_lr_original_0sub = lr_original_0sub.predict(standardised_X_validate_0)
pred_lr_original_meanSub = lr_original_meanSub.predict(standardised_X_validate_mean)
pred_lr_polynomial_0sub = lr_polynomial_0sub.predict(standardised_X_validate_0_poly)
pred_lr_polynomial_meanSub = lr_polynomial_meanSub.predict(standardised_X_validate_mean_poly)
pred_ann_original_0sub = ann_original_0sub.predict(standardised_X_validate_0)
pred_ann_original_meanSub = ann_original_meanSub.predict(standardised_X_validate_mean)
pred_ann_polynomial_0sub = ann_polynomial_0sub.predict(standardised_X_validate_0_poly)
pred_ann_polynomial_meanSub = ann_polynomial_meanSub.predict(standardised_X_validate_mean_poly)

print('Model evaluation metrics: ')
print('Accuracy:')
from sklearn.metrics import accuracy_score

# Logistic Regression
print('Logistic Regression:')
print('Accuracy score - pred_lr_original_0sub: ')
print(accuracy_score(pred_lr_original_0sub, y_validate_0))
print('Accuracy score - pred_lr_original_meanSub: ')
print(accuracy_score(pred_lr_original_meanSub, y_validate_mean))
print('Accuracy score - pred_lr_polynomial_0sub: ')
print(accuracy_score(pred_lr_polynomial_0sub, y_validate_0))
print('Accuracy score - pred_lr_polynomial_meanSub: ')
print(accuracy_score(pred_lr_polynomial_meanSub, y_validate_mean))

# Neural Network
print('Artificial Neural Networks:')
print('Accuracy score - pred_ann_original_0sub: ')
print(accuracy_score(pred_ann_original_0sub, y_validate_0))
print('Accuracy score - pred_ann_original_meanSub: ')
print(accuracy_score(pred_ann_original_meanSub, y_validate_mean))
print('Accuracy score - pred_ann_polynomial_0sub: ')
print(accuracy_score(pred_ann_polynomial_0sub, y_validate_0))
print('Accuracy score - pred_ann_polynomial_meanSub: ')
print(accuracy_score(pred_ann_polynomial_meanSub, y_validate_mean))

print('Classification Report:')
from sklearn.metrics import classification_report

target_names = ['fatality', 'survivor']

# Logistic Regression
print('Logistic Regression:')
print('Classification report - pred_lr_original_0sub: ')
print(classification_report(pred_lr_original_0sub, y_validate_0, target_names=target_names))
print('Classification report - pred_lr_original_meanSub: ')
print(classification_report(pred_lr_original_meanSub, y_validate_mean, target_names=target_names))
print('Classification report - pred_lr_polynomial_0sub: ')
print(classification_report(pred_lr_polynomial_0sub, y_validate_0, target_names=target_names))
print('Classification report - pred_lr_polynomial_meanSub: ')
print(classification_report(pred_lr_polynomial_meanSub, y_validate_mean, target_names=target_names))

# Neural Network
print('Artificial Neural Networks:')
print('Classification report - pred_ann_original_0sub: ')
print(classification_report(pred_ann_original_0sub, y_validate_0, target_names=target_names))
print('Classification report - pred_ann_original_meanSub: ')
print(classification_report(pred_ann_original_meanSub, y_validate_mean, target_names=target_names))
print('Classification report - pred_ann_polynomial_0sub: ')
print(classification_report(pred_ann_polynomial_0sub, y_validate_0, target_names=target_names))
print('Classification report - pred_ann_polynomial_meanSub: ')
print(classification_report(pred_ann_polynomial_meanSub, y_validate_mean, target_names=target_names))

# Step 7 - Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Logistic Regression
parameters = {'C': np.arange(0.1, 4, 0.1),
              'solver': ('liblinear', 'lbfgs'),
              'tol': np.arange(0.00001, 0.001, 0.00001)
              }
gridSearch_lr_0_1 = GridSearchCV(LogisticRegression(max_iter=500), parameters, verbose=1, n_jobs=3)
gridSearch_lr_0_1.fit(standardised_X_train_0, y_train_0)
gridSearch_lr_mean_1 = GridSearchCV(LogisticRegression(max_iter=500), parameters, verbose=1, n_jobs=3)
gridSearch_lr_mean_1.fit(standardised_X_train_mean, y_train_mean)
gridSearch_lr_0_poly = GridSearchCV(LogisticRegression(max_iter=500), parameters, verbose=1, n_jobs=3)
gridSearch_lr_0_poly.fit(standardised_X_train_0_poly, y_train_0)
gridSearch_lr_mean_poly = GridSearchCV(LogisticRegression(max_iter=500), parameters, verbose=1, n_jobs=3)
gridSearch_lr_mean_poly.fit(standardised_X_train_mean_poly, y_train_mean)
print('gridSearch_lr_0_1.best_score_:')
print(gridSearch_lr_0_1.best_score_)
print('gridSearch_lr_0_1.best_params_:')
print(gridSearch_lr_0_1.best_params_)
print('gridSearch_lr_mean_1.best_score_:')
print(gridSearch_lr_mean_1.best_score_)
print('gridSearch_lr_mean_1.best_params_:')
print(gridSearch_lr_mean_1.best_params_)
print('gridSearch_lr_0_poly.best_score_:')
print(gridSearch_lr_0_poly.best_score_)
print('gridSearch_lr_0_poly.best_params_:')
print(gridSearch_lr_0_poly.best_params_)
print('gridSearch_lr_mean_poly.best_score_:')
print(gridSearch_lr_mean_poly.best_score_)
print('gridSearch_lr_mean_poly.best_params_:')
print(gridSearch_lr_mean_poly.best_params_)

# Artificial Neural Network
parameters = {'hidden_layer_sizes': [(100, 50, 25, 12),
                                     (100, 100, 50, 50, 25, 25, 12, 12),
                                     (175, 150, 125, 100, 75, 50, 25, 12),
                                     (2100, 1575, 1050, 525, 250, 175, 100, 50, 25)],
              'activation': ('identity', 'logistic', 'tanh', 'relu'),
              'solver': ('lbfgs', 'sgd', 'adam'),
              'alpha': np.arange(0.00001, 0.0004, 0.00001),
              'learning_rate': ('constant', 'invscaling', 'adaptive'),
              'learning_rate_init': np.arange(0.0001, 0.004, 0.0001),
              'power_t': np.arange(0.1, 1, 0.1),
              'tol': np.arange(0.00001, 0.001, 0.00001)
              }
gridSearch_ann_0_1 = GridSearchCV(MLPClassifier(max_iter=500), parameters, verbose=1, n_jobs=3)
gridSearch_ann_0_1.fit(standardised_X_train_0)
gridSearch_ann_mean_1 = GridSearchCV(MLPClassifier(max_iter=500), parameters, verbose=1, n_jobs=3)
gridSearch_ann_mean_1.fit(standardised_X_train_mean)
gridSearch_ann_0_poly = GridSearchCV(MLPClassifier(max_iter=500), parameters, verbose=1, n_jobs=3)
gridSearch_ann_0_poly.fit(standardised_X_train_0_poly)
gridSearch_ann_mean_poly = GridSearchCV(MLPClassifier(max_iter=500), parameters, verbose=1, n_jobs=3)
gridSearch_ann_mean_poly.fit(standardised_X_train_mean_poly)
print('gridSearch_ann_0_1.best_score_:')
print(gridSearch_ann_0_1.best_score_)
print('gridSearch_ann_0_1.best_params_:')
print(gridSearch_ann_0_1.best_params_)
print('gridSearch_ann_mean_1.best_score_:')
print(gridSearch_ann_mean_1.best_score_)
print('gridSearch_ann_mean_1.best_params_:')
print(gridSearch_ann_mean_1.best_params_)
print('gridSearch_ann_0_poly.best_score_:')
print(gridSearch_ann_0_poly.best_score_)
print('gridSearch_ann_0_poly.best_params_:')
print(gridSearch_ann_0_poly.best_params_)
print('gridSearch_ann_mean_poly.best_score_:')
print(gridSearch_ann_mean_poly.best_score_)
print('gridSearch_ann_mean_poly.best_params_:')
print(gridSearch_ann_mean_poly.best_params_)
exit(code=3)
# Step 8 - Prediction
# Create submittable CSVs
# Get rid of NaNs in the test data:
test_nan_index = np.argwhere(np.isnan(test_data_0))
test_data_0[test_nan_index[0, 0], test_nan_index[0, 1]] = 0
test_data_mean[test_nan_index[0, 0], test_nan_index[0, 1]] = 0
# Create the polynomial versions
test_data_0_poly = poly_2_0.transform(test_data_0)
test_data_mean_poly = poly_2_mean.transform(test_data_mean)
# Standardise all 4 test datasets
standardised_X_test_0 = scaler_0.transform(test_data_0)
standardised_X_test_mean = scaler_mean.transform(test_data_mean)
standardised_X_test_0_poly = scaler_0_poly.transform(test_data_0_poly)
standardised_X_test_mean_poly = scaler_mean_poly.transform(test_data_mean_poly)
# Make the 8 predictions
X_submission_1_0_lr_prediction = lr_original_0sub.predict(standardised_X_test_0)
X_submission_1_mean_lr_prediction = lr_original_meanSub.predict(standardised_X_test_mean)
X_submission_2_0_lr_prediction = lr_polynomial_0sub.predict(standardised_X_test_0_poly)
X_submission_2_mean_lr_prediction = lr_polynomial_meanSub.predict(standardised_X_test_mean_poly)
X_submission_1_0_ann_prediction = ann_original_0sub.predict(standardised_X_test_0)
X_submission_1_mean_ann_prediction = ann_original_meanSub.predict(standardised_X_test_mean)
X_submission_2_0_ann_prediction = ann_polynomial_0sub.predict(standardised_X_test_0_poly)
X_submission_2_mean_ann_prediction = ann_polynomial_meanSub.predict(standardised_X_test_mean_poly)

# Step 9 - Deployment to production
# Save to disk in appropriate format
submission_1_0_lr = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": X_submission_1_0_lr_prediction})
submission_1_mean_lr = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": X_submission_1_mean_lr_prediction})
submission_2_0_lr = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": X_submission_2_0_lr_prediction})
submission_2_mean_lr = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": X_submission_2_mean_lr_prediction})
submission_1_0_ann = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": X_submission_1_0_ann_prediction})
submission_1_mean_ann = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": X_submission_1_mean_ann_prediction})
submission_2_0_ann = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": X_submission_2_0_ann_prediction})
submission_2_mean_ann = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": X_submission_2_mean_ann_prediction})

submission_1_0_lr.to_csv("../datasets/submissions/X_submission_1_0_lr_prediction.csv", index=False)
submission_1_mean_lr.to_csv("../datasets/submissions/X_submission_1_mean_lr_prediction.csv", index=False)
submission_2_0_lr.to_csv("../datasets/submissions/X_submission_2_0_lr_prediction.csv", index=False)
submission_2_mean_lr.to_csv("../datasets/submissions/X_submission_2_mean_lr_prediction.csv", index=False)
submission_1_0_ann.to_csv("../datasets/submissions/X_submission_1_0_ann_prediction.csv", index=False)
submission_1_mean_ann.to_csv("../datasets/submissions/X_submission_1_mean_ann_prediction.csv", index=False)
submission_2_0_ann.to_csv("../datasets/submissions/X_submission_2_0_ann_prediction.csv", index=False)
submission_2_mean_ann.to_csv("../datasets/submissions/X_submission_2_mean_ann_prediction.csv", index=False)
