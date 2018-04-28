#! /usr/bin/python3
# Training a model to correctly identify 8x8 0-16 greyscale images of
# the numerical digits 0-9 as created by human handwriting.

# Step 3 - EDA
from sklearn import datasets

digits = datasets.load_digits()

data = digits.data
target = digits.target

print('data.shape')
print(str(data.shape))
print('target.shape')
print(str(target.shape))

from scipy import stats

stat_desc = stats.describe(data)

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
# 3rd degree polynomials
poly_3 = PolynomialFeatures(3)
poly_3.fit(standardised_X)
X_train_poly_3 = poly_3.transform(standardised_X)
X_test_poly_3 = poly_3.transform(standardised_X_test)

# Step 5 - Model creation and training
# Logistic regression
from sklearn.linear_model import LogisticRegression

lr1 = LogisticRegression(max_iter=500, solver='lbfgs')
lr2 = LogisticRegression(max_iter=500, solver='lbfgs')
lr3 = LogisticRegression(max_iter=500, solver='lbfgs')

lr1.fit(standardised_X, y_train)
lr2.fit(X_train_poly_2, y_train)
lr3.fit(X_train_poly_3, y_train)

y_lr_pred_1 = lr1.predict(standardised_X_test)
y_lr_pred_2 = lr2.predict(X_test_poly_2)
y_lr_pred_3 = lr3.predict(X_test_poly_3)

# Support vector machines
from sklearn.svm import SVC

svc_linear_1 = SVC(kernel='linear', max_iter=500)
svc_linear_2 = SVC(kernel='linear', max_iter=500)
svc_linear_3 = SVC(kernel='linear', max_iter=500)

svc_rbf_1 = SVC(kernel='rbf', max_iter=500)
svc_rbf_2 = SVC(kernel='rbf', max_iter=500)
svc_rbf_3 = SVC(kernel='rbf', max_iter=500)

svc_sigmoid_1 = SVC(kernel='sigmoid', max_iter=500)
svc_sigmoid_2 = SVC(kernel='sigmoid', max_iter=500)
svc_sigmoid_3 = SVC(kernel='sigmoid', max_iter=500)

svc_linear_1.fit(standardised_X, y_train)
svc_linear_2.fit(X_train_poly_2, y_train)
svc_linear_3.fit(X_train_poly_3, y_train)

svc_rbf_1.fit(standardised_X, y_train)
svc_rbf_2.fit(X_train_poly_2, y_train)
svc_rbf_3.fit(X_train_poly_3, y_train)

svc_sigmoid_1.fit(standardised_X, y_train)
svc_sigmoid_2.fit(X_train_poly_2, y_train)
svc_sigmoid_3.fit(X_train_poly_3, y_train)

y_svc_linear_pred_1 = svc_linear_1.predict(standardised_X_test)
y_svc_linear_pred_2 = svc_linear_2.predict(X_test_poly_2)
y_svc_linear_pred_3 = svc_linear_3.predict(X_test_poly_3)

y_svc_rbf_pred_1 = svc_rbf_1.predict(standardised_X_test)
y_svc_rbf_pred_2 = svc_rbf_2.predict(X_test_poly_2)
y_svc_rbf_pred_3 = svc_rbf_3.predict(X_test_poly_3)

y_svc_sigmoid_pred_1 = svc_sigmoid_1.predict(standardised_X_test)
y_svc_sigmoid_pred_2 = svc_sigmoid_2.predict(X_test_poly_2)
y_svc_sigmoid_pred_3 = svc_sigmoid_3.predict(X_test_poly_3)

# Artificial Neural Networks - Multi-layer perceptrons
from sklearn.neural_network import MLPClassifier

mlp_relu_1 = MLPClassifier(activation='relu', max_iter=500)
mlp_relu_2 = MLPClassifier(activation='relu', max_iter=500)
mlp_relu_3 = MLPClassifier(activation='relu', max_iter=500)

mlp_logistic_1 = MLPClassifier(activation='logistic', max_iter=500)
mlp_logistic_2 = MLPClassifier(activation='logistic', max_iter=500)
mlp_logistic_3 = MLPClassifier(activation='logistic', max_iter=500)

mlp_relu_1.fit(standardised_X, y_train)
mlp_relu_2.fit(X_train_poly_2, y_train)
mlp_relu_3.fit(X_train_poly_3, y_train)

mlp_logistic_1.fit(standardised_X, y_train)
mlp_logistic_2.fit(X_train_poly_2, y_train)
mlp_logistic_3.fit(X_train_poly_3, y_train)

y_mlp_relu_pred_1 = mlp_relu_1.predict(standardised_X_test)
y_mlp_relu_pred_2 = mlp_relu_2.predict(X_test_poly_2)
y_mlp_relu_pred_3 = mlp_relu_3.predict(X_test_poly_3)

y_mlp_logistic_pred_1 = mlp_logistic_1.predict(standardised_X_test)
y_mlp_logistic_pred_2 = mlp_logistic_2.predict(X_test_poly_2)
y_mlp_logistic_pred_3 = mlp_logistic_3.predict(X_test_poly_3)

# Step 6 - model evaluation
from sklearn.metrics import accuracy_score, classification_report

# Logistic regression
print('Accuracy score, Linear Regression, 1st order features:')
print(accuracy_score(y_test, y_lr_pred_1))
print('Accuracy score, Linear Regression, 2nd order features:')
print(accuracy_score(y_test, y_lr_pred_2))
print('Accuracy score, Linear Regression, 3rd order features:')
print(accuracy_score(y_test, y_lr_pred_3))

print('Classification report, Linear Regression, 1st order features:')
print(classification_report(y_test, y_lr_pred_1))
print('Classification report, Linear Regression, 2nd order features:')
print(classification_report(y_test, y_lr_pred_2))
print('Classification report, Linear Regression, 3rd order features:')
print(classification_report(y_test, y_lr_pred_3))

# Support vector machines
# Linear
print('Accuracy score, SVM, Linear kernel, 1st order features:')
print(accuracy_score(y_test, y_svc_linear_pred_1))
print('Accuracy score, SVM, Linear kernel, 2nd order features:')
print(accuracy_score(y_test, y_svc_linear_pred_2))
print('Accuracy score, SVM, Linear kernel, 3rd order features:')
print(accuracy_score(y_test, y_svc_linear_pred_3))

print('Classification report, SVM, Linear kernel, 1st order features:')
print(classification_report(y_test, y_svc_linear_pred_1))
print('Classification report, SVM, Linear kernel, 2nd order features:')
print(classification_report(y_test, y_svc_linear_pred_2))
print('Classification report, SVM, Linear kernel, 3rd order features:')
print(classification_report(y_test, y_svc_linear_pred_3))

# RBF
print('Accuracy score, SVM, RBF kernel, 1st order features:')
print(accuracy_score(y_test, y_svc_rbf_pred_1))
print('Accuracy score, SVM, RBF kernel, 2nd order features:')
print(accuracy_score(y_test, y_svc_rbf_pred_2))
print('Accuracy score, SVM, RBF kernel, 3rd order features:')
print(accuracy_score(y_test, y_svc_rbf_pred_3))

print('Classification report, SVM, Linear RBF, 1st order features:')
print(classification_report(y_test, y_svc_rbf_pred_1))
print('Classification report, SVM, Linear RBF, 2nd order features:')
print(classification_report(y_test, y_svc_rbf_pred_2))
print('Classification report, SVM, Linear RBF, 3rd order features:')
print(classification_report(y_test, y_svc_rbf_pred_3))

# Sigmoid
print('Accuracy score, SVM, Sigmoid kernel, 1st order features:')
print(accuracy_score(y_test, y_svc_sigmoid_pred_1))
print('Accuracy score, SVM, Sigmoid kernel, 2nd order features:')
print(accuracy_score(y_test, y_svc_sigmoid_pred_2))
print('Accuracy score, SVM, Sigmoid kernel, 3rd order features:')
print(accuracy_score(y_test, y_svc_sigmoid_pred_3))

print('Classification report, SVM, Linear sigmoid, 1st order features:')
print(classification_report(y_test, y_svc_sigmoid_pred_1))
print('Classification report, SVM, Linear sigmoid, 2nd order features:')
print(classification_report(y_test, y_svc_sigmoid_pred_2))
print('Classification report, SVM, Linear sigmoid, 3rd order features:')
print(classification_report(y_test, y_svc_sigmoid_pred_3))

# Artificial neural networks - Multi-layer perceptrons
# relu
print('Accuracy score, ANN, activation: relu, 1st order features:')
print(accuracy_score(y_test, y_mlp_relu_pred_1))
print('Accuracy score, ANN, activation: relu, 2nd order features:')
print(accuracy_score(y_test, y_mlp_relu_pred_2))
print('Accuracy score, ANN, activation: relu, 3rd order features:')
print(accuracy_score(y_test, y_mlp_relu_pred_3))

print('Classification report, ANN, activation: relu, 1st order features:')
print(classification_report(y_test, y_mlp_relu_pred_1))
print('Classification report, ANN, activation: relu, 2nd order features:')
print(classification_report(y_test, y_mlp_relu_pred_2))
print('Classification report, ANN, activation: relu, 3rd order features:')
print(classification_report(y_test, y_mlp_relu_pred_3))

# logistic
print('Accuracy score, ANN, activation: logistic, 1st order features:')
print(accuracy_score(y_test, y_mlp_logistic_pred_1))
print('Accuracy score, ANN, activation: logistic, 2nd order features:')
print(accuracy_score(y_test, y_mlp_logistic_pred_2))
print('Accuracy score, ANN, activation: logistic, 3rd order features:')
print(accuracy_score(y_test, y_mlp_logistic_pred_3))

print('Classification report, ANN, activation: logistic, 1st order features:')
print(classification_report(y_test, y_mlp_logistic_pred_1))
print('Classification report, ANN, activation: logistic, 2nd order features:')
print(classification_report(y_test, y_mlp_logistic_pred_2))
print('Classification report, ANN, activation: logistic, 3rd order features:')
print(classification_report(y_test, y_mlp_logistic_pred_3))

# Save the models somewhere
from sklearn.externals import joblib

joblib.dump(lr1, 'gitIgnoreDir/digits/lr1.pkl')
joblib.dump(lr2, 'gitIgnoreDir/digits/lr2.pkl')
joblib.dump(lr3, 'gitIgnoreDir/digits/lr3.pkl')

joblib.dump(svc_linear_1, 'gitIgnoreDir/digits/svc_linear_1.pkl')
joblib.dump(svc_linear_2, 'gitIgnoreDir/digits/svc_linear_2.pkl')
joblib.dump(svc_linear_3, 'gitIgnoreDir/digits/svc_linear_3.pkl')

joblib.dump(svc_rbf_1, 'gitIgnoreDir/digits/svc_rbf_1.pkl')
joblib.dump(svc_rbf_2, 'gitIgnoreDir/digits/svc_rbf_2.pkl')
joblib.dump(svc_rbf_3, 'gitIgnoreDir/digits/svc_rbf_3.pkl')

joblib.dump(svc_sigmoid_1, 'gitIgnoreDir/digits/svc_sigmoid_1.pkl')
joblib.dump(svc_sigmoid_2, 'gitIgnoreDir/digits/svc_sigmoid_2.pkl')
joblib.dump(svc_sigmoid_3, 'gitIgnoreDir/digits/svc_sigmoid_3.pkl')

joblib.dump(mlp_relu_1, 'gitIgnoreDir/digits/svc_rbf_1.pkl')
joblib.dump(mlp_relu_2, 'gitIgnoreDir/digits/svc_rbf_2.pkl')
joblib.dump(mlp_relu_3, 'gitIgnoreDir/digits/svc_rbf_3.pkl')

joblib.dump(mlp_logistic_1, 'gitIgnoreDir/digits/svc_rbf_1.pkl')
joblib.dump(mlp_logistic_2, 'gitIgnoreDir/digits/svc_rbf_2.pkl')
joblib.dump(mlp_logistic_3, 'gitIgnoreDir/digits/svc_rbf_3.pkl')
