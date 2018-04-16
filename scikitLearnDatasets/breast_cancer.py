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
