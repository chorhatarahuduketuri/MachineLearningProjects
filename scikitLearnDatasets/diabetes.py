#! /usr/bin/python3
# Trying to predict diabetes disease progression

# Step 2
# Load the dataset
from sklearn import datasets

diabetes = datasets.load_diabetes()

data = diabetes.data
target = diabetes.target

# Step 3 - EDA
print('data.shape:')
print(data.shape)
print('target.shape')
print(data.shape)

from scipy import stats

stats.describe(data)

# Step 4 - feature engineering
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
standardised_X = scaler.transform(X_train)
standardised_X_test = scaler.transform(X_test)

from sklearn.preprocessing import PolynomialFeatures

# 2nd degree polynomials
poly_2 = PolynomialFeatures(2)
poly_2.fit(standardised_X)
X_train_poly_2 = poly_2.transform(standardised_X)
X_test_poly_2 = poly_2.transform(standardised_X_test)

# Step 5 - Model creation and training
# Linear Regression
from sklearn.linear_model import LinearRegression

lr1 = LinearRegression()
lr2 = LinearRegression()

lr1.fit(standardised_X, y_train)
lr2.fit(X_train_poly_2, y_train)

lr1_pred = lr1.predict(standardised_X_test)
lr2_pred = lr2.predict(X_test_poly_2)

# Support Vector Machine
from sklearn.svm import SVR

svr1 = SVR()
svr2 = SVR()

svr1.fit(standardised_X, y_train)
svr2.fit(X_train_poly_2, y_train)

svr1_pred = svr1.predict(standardised_X_test)
svr2_pred = svr2.predict(X_test_poly_2)

# Artificial Neural Network
from sklearn.neural_network import MLPRegressor

ann1 = MLPRegressor(max_iter=5000, hidden_layer_sizes=(10, 10))
ann2 = MLPRegressor(max_iter=5000, hidden_layer_sizes=(66, 66))

ann1.fit(standardised_X, y_train)
ann2.fit(X_train_poly_2, y_train)

ann_1_pred = ann1.predict(standardised_X_test)
ann_2_pred = ann2.predict(X_test_poly_2)

# Step 6 - model evaluation
from sklearn.metrics import mean_absolute_error

print('Mean absolute error, logistic regression, 1st order training data: ')
print(mean_absolute_error(y_test, lr1_pred))
print('Mean absolute error, logistic regression, 2nd order training data: ')
print(mean_absolute_error(y_test, lr2_pred))
print('Mean absolute error, SVM, 1st order training data: ')
print(mean_absolute_error(y_test, svr1_pred))
print('Mean absolute error, SVM, 2nd order training data: ')
print(mean_absolute_error(y_test, svr2_pred))
print('Mean absolute error, ANN, 1st order training data: ')
print(mean_absolute_error(y_test, ann_1_pred))
print('Mean absolute error, ANN, 2nd order training data: ')
print(mean_absolute_error(y_test, ann_2_pred))

# Save the models somewhere
from sklearn.externals import joblib

joblib.dump(lr1, 'gitIgnoreDir/diabetes/lr_default_1.pkl')
joblib.dump(lr2, 'gitIgnoreDir/diabetes/lr_default_2.pkl')
joblib.dump(svr1, 'gitIgnoreDir/diabetes/svr_default_1.pkl')
joblib.dump(svr2, 'gitIgnoreDir/diabetes/svr_default_2.pkl')
joblib.dump(ann1, 'gitIgnoreDir/diabetes/ann_1.pkl')
joblib.dump(ann2, 'gitIgnoreDir/diabetes/ann_2.pkl')

# Repeat 100 times and summarise: -------------------------------------------------------------------------
import numpy as np

iterations = 100
errs = np.zeros((6, iterations))
for i in range(iterations):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    scaler = StandardScaler().fit(X_train)
    standardised_X = scaler.transform(X_train)
    standardised_X_test = scaler.transform(X_test)
    poly_2 = PolynomialFeatures(2)
    poly_2.fit(standardised_X)
    X_train_poly_2 = poly_2.transform(standardised_X)
    X_test_poly_2 = poly_2.transform(standardised_X_test)
    lr1 = LinearRegression()
    lr2 = LinearRegression()
    lr1.fit(standardised_X, y_train)
    lr2.fit(X_train_poly_2, y_train)
    lr1_pred = lr1.predict(standardised_X_test)
    lr2_pred = lr2.predict(X_test_poly_2)
    svr1 = SVR()
    svr2 = SVR()
    svr1.fit(standardised_X, y_train)
    svr2.fit(X_train_poly_2, y_train)
    svr1_pred = svr1.predict(standardised_X_test)
    svr2_pred = svr2.predict(X_test_poly_2)
    ann1 = MLPRegressor(max_iter=5000, hidden_layer_sizes=(10, 10))
    ann2 = MLPRegressor(max_iter=5000, hidden_layer_sizes=(66, 66))
    ann1.fit(standardised_X, y_train)
    ann2.fit(X_train_poly_2, y_train)
    ann_1_pred = ann1.predict(standardised_X_test)
    ann_2_pred = ann2.predict(X_test_poly_2)
    errs[0, i] = mean_absolute_error(y_test, lr1_pred)
    errs[1, i] = mean_absolute_error(y_test, lr2_pred)
    errs[2, i] = mean_absolute_error(y_test, svr1_pred)
    errs[3, i] = mean_absolute_error(y_test, svr2_pred)
    errs[4, i] = mean_absolute_error(y_test, ann_1_pred)
    errs[5, i] = mean_absolute_error(y_test, ann_2_pred)

print(stats.describe(errs, axis=1))
