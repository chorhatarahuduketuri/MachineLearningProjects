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