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
