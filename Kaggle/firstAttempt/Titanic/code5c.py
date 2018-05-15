# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Data loading
training_data = pd.read_csv('train.csv')

# Data preprocessing and putting into a data structure that can be
# used by a classifier
training_data["Age"] = training_data["Age"].fillna(training_data["Age"].median())
training_data["Sex"].loc[training_data["Sex"] == "male"] = 0
training_data["Sex"].loc[training_data["Sex"] == "female"] = 1
training_data["Embarked"] = training_data["Embarked"].fillna(0)
training_data["Embarked"].loc[training_data["Embarked"] == "C"] = 1
training_data["Embarked"].loc[training_data["Embarked"] == "Q"] = 2
training_data["Embarked"].loc[training_data["Embarked"] == "S"] = 3

X = training_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = training_data["Survived"]

# Somewhere to put the results
results = np.zeros([1000, 7])

# Some useful objects for the training loop
n_components = list(range(1, 8))
pca = PCA()
lr = LogisticRegression(max_iter=500, solver='lbfgs')

# Run the training 1000 times, store results in results
for i in range(0, 1000):
    for n in n_components:
        pca.set_params(n_components=n)
        pca.fit(X, y)
        X_PCA = pca.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_PCA, y, test_size=0.8)
        lr.fit(X_train, y_train)
        results[i][n - 1] = lr.score(X_test, y_test)

# Take get averages of results
results_average = np.average(results, 0)

plt.figure(1, figsize=(10, 10))
plt.clf()
plt.xlabel('n_components')
plt.ylabel('test set accuracy, avg. over 1000 runs')
plt.plot(n_components, results_average)
plt.title('Average test set accuracy per n_components')
plt.savefig('submissions_and_results/code5c1.png', bbox_inches='tight')

plt.figure(2, figsize=(10, 10))
plt.clf()
plt.xlabel('n_components')
plt.ylabel('test set accuracy, avg. over 1000 runs')
plt.plot(n_components, results_average)
plt.title('Average test set accuracy per n_components')
plt.yticks(results_average)
plt.savefig('submissions_and_results/code5c2.png', bbox_inches='tight')
