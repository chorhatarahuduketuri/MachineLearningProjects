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
ax1.hist(data[:, 4], 250, histtype='bar', label=feature_name_CRIM)
ax1.legend()
ax1.set(title='Distribution of the ' + feature_name_CRIM + ' data',
        ylabel='Number of data points',
        xlabel=feature_name_CRIM + ' measurements')
fig1.savefig('gitIgnoreDir/boston/' + feature_name_CRIM + '.png')
