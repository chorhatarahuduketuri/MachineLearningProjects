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
