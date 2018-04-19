#! /usr/bin/python3
# An analysis of the Iris dataset, beginning with an EDA.

# Step 3 - EDA
# Load the Iris dataset
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
target = iris.target

# Show the data
print('Feature data:')
print(data)
print('Target data:')
print(target)

# Show the shape of the data, which tells us how much there is
print('Feature data shape:')
print(str(data.shape))
print('Target data shape:')
print(str(target.shape))

# Get basic statistical information about the data
from scipy import stats

stats.describe(data)

# Make graphs about the data
import numpy as np
import matplotlib.pyplot as plt

# Get data to be graphed
sepalLength = np.array([data[0:50, 0], data[50:100, 0], data[100:150, 0]]).transpose()
sepalWidth = np.array([data[0:50, 1], data[50:100, 1], data[100:150, 1]]).transpose()
petalLength = np.array([data[0:50, 2], data[50:100, 2], data[100:150, 2]]).transpose()
petalWidth = np.array([data[0:50, 3], data[50:100, 3], data[100:150, 3]]).transpose()

# Create plots
# sepalLength
figSepLen = plt.figure()
axSepLen = figSepLen.add_subplot()
figSepLen, axSepLen = plt.subplots()
axSepLen.hist(sepalLength, 20, histtype='bar', label=('Setosa', 'Versicolor', 'Virginica'))
axSepLen.legend()
axSepLen.set(title='Comparison of the Sepal lengths of the three Iris flower types',
             ylabel='Number of data points',
             xlabel='Sepal lengths')
plt.xticks(np.arange(4, 8.01, step=0.5))

# sepalWidth
figSepWid = plt.figure()
axSepWid = figSepWid.add_subplot()
figSepWid, axSepWid = plt.subplots()
axSepWid.hist(sepalWidth, 20, histtype='bar', label=('Setosa', 'Versicolor', 'Virginica'))
axSepWid.legend()
axSepWid.set(title='Comparison of the Sepal widths of the three Iris flower types',
             ylabel='Number of data points',
             xlabel='Sepal widths')
plt.xticks(np.arange(2, 5, step=0.5))

# petalLength
figPetLen = plt.figure()
axPetLen = figPetLen.add_subplot()
figPetLen, axPetLen = plt.subplots()
axPetLen.hist(petalLength, 20, histtype='bar', label=('Setosa', 'Versicolor', 'Virginica'))
axPetLen.legend()
axPetLen.set(title='Comparison of the Petal lengths of the three Iris flower types',
             ylabel='Number of data points',
             xlabel='Petal lengths')
plt.xticks(np.arange(1, 7, step=0.5))

# petalWidth
figPetWid = plt.figure()
axPetWid = figPetWid.add_subplot()
figPetWid, axPetWid = plt.subplots()
axPetWid.hist(petalWidth, 20, histtype='bar', label=('Setosa', 'Versicolor', 'Virginica'))
axPetWid.legend()
axPetWid.set(title='Comparison of the Petal widths of the three Iris flower types',
             ylabel='Number of data points',
             xlabel='Petal widths')
plt.xticks(np.arange(0, 2.51, step=0.5))

figSepLen.savefig('gitIgnoreDir/sepalLength.png')
figSepWid.savefig('gitIgnoreDir/sepalWidth.png')
figPetLen.savefig('gitIgnoreDir/petalLength.png')
figPetWid.savefig('gitIgnoreDir/petalWidth.png')

# Step 4 - Model selection and feature engineering
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
standardised_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)

from sklearn.preprocessing import PolynomialFeatures

# 2nd degree polynomials
poly_2 = PolynomialFeatures(2)
poly_2.fit(X_train)
X_train_poly_2 = poly_2.transform(X_train)
X_test_poly_2 = poly_2.transform(X_test)
# 3rd degree polynomials
poly_3 = PolynomialFeatures(3)
poly_3.fit(X_train)
X_train_poly_3 = poly_3.transform(X_train)
X_test_poly_3 = poly_3.transform(X_test)

# Step 5 - Model creation and training
# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr1 = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=500)
lr2 = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=500)
lr3 = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=500)

lr1.fit(X_train, y_train)
lr2.fit(X_train_poly_2, y_train)
lr3.fit(X_train_poly_3, y_train)

y_lr_pred_1 = lr1.predict(X_test)
y_lr_pred_2 = lr2.predict(X_test_poly_2)
y_lr_pred_3 = lr3.predict(X_test_poly_3)

from sklearn.metrics import accuracy_score

print('Logistic Regression: ')
print(accuracy_score(y_test, y_lr_pred_1))
print(accuracy_score(y_test, y_lr_pred_2))
print(accuracy_score(y_test, y_lr_pred_3))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_lr_pred_1))
print(classification_report(y_test, y_lr_pred_2))
print(classification_report(y_test, y_lr_pred_3))

# Support Vector Machine
from sklearn.svm import SVC

svc1 = SVC(max_iter=500)
svc2 = SVC(max_iter=500)
svc3 = SVC(max_iter=500)

svc1.fit(X_train, y_train)
svc2.fit(X_train_poly_2, y_train)
svc3.fit(X_train_poly_3, y_train)

y_svc_pred_1 = svc1.predict(X_test)
y_svc_pred_2 = svc2.predict(X_test_poly_2)
y_svc_pred_3 = svc3.predict(X_test_poly_3)

print('Support Vector Machine: ')
print(accuracy_score(y_test, y_svc_pred_1))
print(accuracy_score(y_test, y_svc_pred_2))
print(accuracy_score(y_test, y_svc_pred_3))
print(classification_report(y_test, y_svc_pred_1))
print(classification_report(y_test, y_svc_pred_2))
print(classification_report(y_test, y_svc_pred_3))

from sklearn.externals import joblib

joblib.dump(lr1, 'gitIgnoreDir/lr1.pkl')
joblib.dump(lr2, 'gitIgnoreDir/lr2.pkl')
joblib.dump(lr3, 'gitIgnoreDir/lr3.pkl')

joblib.dump(svc1, 'gitIgnoreDir/svc1.pkl')
joblib.dump(svc2, 'gitIgnoreDir/svc2.pkl')
joblib.dump(svc3, 'gitIgnoreDir/svc3.pkl')
