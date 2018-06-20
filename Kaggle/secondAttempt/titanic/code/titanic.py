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
training_data_0 = titanic_training[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].as_matrix()
nan_indicies = np.isnan(training_data_0[:, 2])
training_data_0[:, 2][nan_indicies] = 0
# meansub
training_data_mean = titanic_training[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].as_matrix()
training_age_mean = np.nanmean(training_data_mean, axis=0)[2]
training_data_mean[:, 2][nan_indicies] = training_age_mean
# testing
# 0sub
test_data_0 = titanic_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].as_matrix()
nan_indicies = np.isnan(test_data_0[:, 1])
test_data_0[:, 2][nan_indicies] = 0
# meansub
test_data_mean = titanic_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].as_matrix()
test_age_mean = np.nanmean(test_data_mean, axis=0)[1]
test_data_mean[:, 2][nan_indicies] = test_age_mean

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
