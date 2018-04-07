#! /usr/bin/python3
# An analysis of the Iris dataset, beginning with an EDA.

# Step 3 - EDA
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
target = iris.target

print('Feature data:')
print(data)
print('Target data:')
print(target)

print('Feature data shape:')
print(str(data.shape))
print('Target data shape:')
print(str(target.shape))

from scipy import stats
stats.describe(data)
