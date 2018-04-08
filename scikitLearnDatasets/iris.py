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

