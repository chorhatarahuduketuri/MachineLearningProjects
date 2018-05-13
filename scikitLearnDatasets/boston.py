#! /usr/bin/python3
# An analysis and training of a model on the boston housing prices dataset

from sklearn import datasets

boston = datasets.load_boston()

data = boston.data
target = boston.target

# Step 3
print('data.shape:')
print(data.shape)
print('target.shape')
print(data.shape)

from scipy import stats

stats.describe(data)

import matplotlib.pyplot as plt

fig1 = plt.figure()
ax1 = fig1.add_subplot()
fig1, ax1 = plt.subplots()
feature_name_CRIM = boston.feature_names[0]
ax1.hist(data[:, 0], 250, histtype='bar', label=feature_name_CRIM)
ax1.legend()
ax1.set(title='Distribution of the ' + feature_name_CRIM + ' data',
        ylabel='Number of data points',
        xlabel=feature_name_CRIM + ' measurements')
fig1.savefig('gitIgnoreDir/boston/' + feature_name_CRIM + '.png')

fig2 = plt.figure()
ax2 = fig2.add_subplot()
fig2, ax2 = plt.subplots()
feature_name_NOX = boston.feature_names[4]
ax2.hist(data[:, 4], 250, histtype='bar', label=feature_name_NOX)
ax2.legend()
ax2.set(title='Distribution of the ' + feature_name_NOX + ' data',
        ylabel='Number of data points',
        xlabel=feature_name_NOX + ' measurements')
fig2.savefig('gitIgnoreDir/boston/' + feature_name_NOX + '.png')

# Step 4 - Model selection and feature engineering
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

# Step 6 - model evaluation
from sklearn.metrics import mean_absolute_error

print('Mean absolute error, logistic regression, 1st order training data, in steps of $1000: ')
print(mean_absolute_error(y_test, lr1_pred))
print('Mean absolute error, logistic regression, 2nd order training data, in steps of $1000: ')
print(mean_absolute_error(y_test, lr2_pred))
print('Mean absolute error, SVM, 1st order training data, in steps of $1000: ')
print(mean_absolute_error(y_test, svr1_pred))
print('Mean absolute error, SVM, 2nd order training data, in steps of $1000: ')
print(mean_absolute_error(y_test, svr2_pred))

# Step 5 - take 2
from sklearn.neural_network import MLPRegressor

ann1 = MLPRegressor(max_iter=5000, hidden_layer_sizes=(13, 13))
ann2 = MLPRegressor(max_iter=5000, hidden_layer_sizes=(105, 105))

ann1.fit(standardised_X, y_train)
ann2.fit(X_train_poly_2, y_train)

ann_1_pred = ann1.predict(standardised_X_test)
ann_2_pred = ann2.predict(X_test_poly_2)

print('Mean absolute error, ANN, 1st order training data, in steps of $1000: ')
print(mean_absolute_error(y_test, ann_1_pred))
print('Mean absolute error, ANN, 2nd order training data, in steps of $1000: ')
print(mean_absolute_error(y_test, ann_2_pred))

# Save the models somewhere
from sklearn.externals import joblib

joblib.dump(lr1, 'gitIgnoreDir/boston/lr_default_1.pkl')
joblib.dump(lr2, 'gitIgnoreDir/boston/lr_default_2.pkl')
joblib.dump(svr1, 'gitIgnoreDir/boston/svr_default_1.pkl')
joblib.dump(svr2, 'gitIgnoreDir/boston/svr_default_2.pkl')
joblib.dump(ann1, 'gitIgnoreDir/boston/ann_1.pkl')
joblib.dump(ann2, 'gitIgnoreDir/boston/ann_2.pkl')
