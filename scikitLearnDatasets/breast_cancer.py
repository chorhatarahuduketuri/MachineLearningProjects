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
