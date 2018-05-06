#! /usr/bin/python3
# An analysis of some wine data, plus the training of a machine
# learning model to identify different types of wine.

# Step 3 - EDA
from sklearn import datasets

wine = datasets.load_wine()

data = wine.data
target = wine.target

print('data.shape')
print(data.shape)
print('target.shape')
print(target.shape)

# Create graph of feature with interesting skew/kurt
import matplotlib.pyplot as plt

fig1 = plt.figure()
ax1 = fig1.add_subplot()
fig1, ax1 = plt.subplots()
feature_name_magnesium = wine.feature_names[4]
ax1.hist(data[:, 4], 90, histtype='bar', label=feature_name_magnesium)
ax1.legend()
ax1.set(title='Distribution of the ' + feature_name_magnesium + ' data',
        ylabel='Number of data points',
        xlabel=feature_name_magnesium + ' measurements')
fig1.savefig('gitIgnoreDir/wine/' + feature_name_magnesium + '.png')

# Step 4 - Model selection and feature engineering
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
standardised_X_train = scaler.transform(X_train)
standardised_X_test = scaler.transform(X_test)

from sklearn.preprocessing import PolynomialFeatures
poly_2 = PolynomialFeatures(2)
poly_2.fit(standardised_X_train)
X_train_poly_2 = poly_2.transform(standardised_X_train)
X_test_poly_2 = poly_2.transform(standardised_X_test)


# Step 5 - Model creation and training
# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr1 = LogisticRegression(max_iter=500, solver='lbfgs')
lr2 = LogisticRegression(max_iter=500, solver='lbfgs')

lr1.fit(standardised_X_train, y_train)
lr2.fit(X_train_poly_2, y_train)

y_lr_pred_1 = lr1.predict(standardised_X_test)
y_lr_pred_2 = lr2.predict(X_test_poly_2)

# SVM
from sklearn.svm import SVC

svc_default_1 = SVC()
svc_default_2 = SVC()

svc_default_1.fit(standardised_X_train, y_train)
svc_default_2.fit(X_train_poly_2, y_train)

y_svc_default_pred_1 = svc_default_1.predict(standardised_X_test)
y_svc_default_pred_2 = svc_default_2.predict(X_test_poly_2)

# Step 6 - model evaluation
from sklearn.metrics import accuracy_score
# Logistic Regression
accuracy_score(y_test, y_lr_pred_1)
accuracy_score(y_test, y_lr_pred_2)

# SVM
accuracy_score(y_test, y_svc_default_pred_1)
accuracy_score(y_test, y_svc_default_pred_2)
