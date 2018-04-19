#! /usr/bin/python3
# Creation of a model to classify human breast tumours as malignant or
# benign by the use of machine learning.

# Step 3 - EDA
# Load the Breast Cancer dataset
from sklearn import datasets

breast_cancer = datasets.load_breast_cancer()
data = breast_cancer.data
target = breast_cancer.target

# Get basic statistical information about the dataset
from scipy import stats

stats.describe(data)

# Graphs
# Assemble data structures to be graphed
means = data[:, 0:10]
errors = data[:, 10:20]
worst = data[:, 20:30]

feature_names = breast_cancer.feature_names
mean_names = feature_names[0:10]
error_names = feature_names[10:20]
worst_names = feature_names[20:30]

# Create plots
import matplotlib.pyplot as plt

# Means
for i in range(0, 10):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    fig1, ax1 = plt.subplots()
    ax1.hist(means[:, i], 100, histtype='bar', label=(mean_names[i]))
    ax1.legend()
    ax1.set(title='Distribution of the ' + mean_names[i] + ' data',
            ylabel='Number of data points',
            xlabel='Mean measurements')
    fig1.savefig('gitIgnoreDir/breast_cancer/histograms/mean/hist_' + mean_names[i].replace(' ', '_') + '.png')

# Errors
for i in range(0, 10):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    fig1, ax1 = plt.subplots()
    ax1.hist(errors[:, i], 100, histtype='bar', label=(error_names[i]))
    ax1.legend()
    ax1.set(title='Distribution of the ' + error_names[i] + ' data',
            ylabel='Number of data points',
            xlabel='Error measurements')
    fig1.savefig('gitIgnoreDir/breast_cancer/histograms/error/hist_' + error_names[i].replace(' ', '_') + '.png')

# Worst
for i in range(0, 10):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    fig1, ax1 = plt.subplots()
    ax1.hist(worst[:, i], 100, histtype='bar', label=(worst_names[i]))
    ax1.legend()
    ax1.set(title='Distribution of the ' + worst_names[i] + ' data',
            ylabel='Number of data points',
            xlabel='Worst measurements')
    fig1.savefig('gitIgnoreDir/breast_cancer/histograms/worst/hist_' + worst_names[i].replace(' ', '_') + '.png')

# Step 4 - Model selection and feature engineering
# Create training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

# Standardise the training and test data (mean normalisation)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Generate 2nd order polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)
poly.fit_transform(X_train)
X_train_poly = poly.transform(X_train)
X_test_poly = poly.transform(X_test)

# Step 5 - Model creation and training
# Logistic regression
from sklearn.linear_model import LogisticRegression

lr1 = LogisticRegression(solver='lbfgs', max_iter=500)
lr2 = LogisticRegression(solver='lbfgs', max_iter=500)

lr1.fit(X_train, y_train)
lr2.fit(X_train_poly, y_train)

y_lr_pred_1 = lr1.predict(X_test)
y_lr_pred_2 = lr2.predict(X_test_poly)

# Support Vector Machines
from sklearn.svm import SVC

svc_linear_1 = SVC(kernel='linear', max_iter=500)
svc_linear_2 = SVC(kernel='linear', max_iter=500)

svc_rbf_1 = SVC(kernel='rbf', max_iter=500)
svc_rbf_2 = SVC(kernel='rbf', max_iter=500)

svc_linear_1.fit(X_train, y_train)
svc_linear_2.fit(X_train_poly, y_train)

svc_rbf_1.fit(X_train, y_train)
svc_rbf_2.fit(X_train_poly, y_train)

y_svc_linear_pred_1 = svc_linear_1.predict(X_test)
y_svc_linear_pred_2 = svc_linear_2.predict(X_test_poly)

y_svc_rbf_pred_1 = svc_rbf_1.predict(X_test)
y_svc_rbf_pred_2 = svc_rbf_2.predict(X_test_poly)

# Step 6 - model evaluation
from sklearn.metrics import accuracy_score, classification_report

# Logistic regression
print('Accuracy Score, Linear Regression, 1st order features: ')
print(accuracy_score(y_test, y_lr_pred_1))
print('Accuracy Score, Linear Regression, 2nd order features: ')
print(accuracy_score(y_test, y_lr_pred_2))

print('Classification Report, Linear Regression, 1st order features: ')
print(classification_report(y_test, y_lr_pred_1))
print('Classification Report, Linear Regression, 2nd order features: ')
print(classification_report(y_test, y_lr_pred_2))

# Suport Vector Machines
# Linear
print('Accuracy Score, SVM, Linear kernel, 1st order features: ')
print(accuracy_score(y_test, y_svc_linear_pred_1))
print('Accuracy Score, SVM, Linear kernel, 2nd order features: ')
print(accuracy_score(y_test, y_svc_linear_pred_2))

print('Classification Report, SVM, Linear kernel, 1st order features: ')
print(classification_report(y_test, y_svc_linear_pred_1))
print('Classification Report, SVM, Linear kernel, 2nd order features: ')
print(classification_report(y_test, y_svc_linear_pred_2))

# RBF
print('Accuracy Score, SVM, RBF kernel, 1st order features: ')
print(accuracy_score(y_test, y_svc_rbf_pred_1))
print('Accuracy Score, SVM, RBF kernel, 2nd order features: ')
print(accuracy_score(y_test, y_svc_rbf_pred_2))

print('Classification Report, SVM, RBF kernel, 1st order features: ')
print(classification_report(y_test, y_svc_rbf_pred_1))
print('Classification Report, SVM, RBF kernel, 2nd order features: ')
print(classification_report(y_test, y_svc_rbf_pred_2))

# Save the models somewhere
from sklearn.externals import joblib

joblib.dump(lr1, 'gitIgnoreDir/breast_cancer/lr1.pkl')
joblib.dump(lr2, 'gitIgnoreDir/breast_cancer/lr2.pkl')

joblib.dump(svc_linear_1, 'gitIgnoreDir/breast_cancer/svc_linear_1.pkl')
joblib.dump(svc_linear_2, 'gitIgnoreDir/breast_cancer/svc_linear_2.pkl')

joblib.dump(svc_rbf_1, 'gitIgnoreDir/breast_cancer/svc_rbf_1.pkl')
joblib.dump(svc_rbf_2, 'gitIgnoreDir/breast_cancer/svc_rbf_2.pkl')
