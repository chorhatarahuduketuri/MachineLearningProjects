# Imports
import numpy as np
import pandas as pd
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

# Generate training and cross validation sets, split two thirds training, one third cross validation
X = training_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
X["PclassSquared"] = X["Pclass"]**2
X["Sex"] = (X["Sex"]+0.5)**2
X["Age"] = X["Age"]**2
X["SibSp"] = X["SibSp"]**2
X["Parch"] = X["Parch"]**2
X["Fare"] = X["Fare"]**2
X["Embarked"] = X["Embarked"]**2
y = training_data["Survived"]
X_train, X_cross_validation, y_train, y_cross_validation = train_test_split(X, y, test_size=(1 / 3.))

# Create classifier with parameters
logRegClasif = LogisticRegression()

# Create ranges of parameters
solvers = ['saga', 'sag', 'liblinear', 'lbfgs', 'newton-cg']
iterations = [100, 200, 300, 400, 500]
Cs = np.linspace(1e-4, 1)

sagaResults = pd.DataFrame(columns=iterations, index=Cs)
sagResults = pd.DataFrame(columns=iterations, index=Cs)
liblinearResults = pd.DataFrame(columns=iterations, index=Cs)
lbfgsResults = pd.DataFrame(columns=iterations, index=Cs)
newtoncgResults = pd.DataFrame(columns=iterations, index=Cs)

# For each combination of parameters (1250 combinations),
# set the classifier to have those parameters, train it, score it,
# and store the results in a dataframe

# saga solver
for numOfIterations in iterations:
    for C in Cs:
        logRegClasif.set_params(max_iter=numOfIterations, solver='saga', C=C)
        logRegClasif.fit(X_train, y_train)
        sagaResults[numOfIterations][C] = logRegClasif.score(X_cross_validation, y_cross_validation)

# sag solver
for numOfIterations in iterations:
    for C in Cs:
        logRegClasif.set_params(max_iter=numOfIterations, solver='sag', C=C)
        logRegClasif.fit(X_train, y_train)
        sagResults[numOfIterations][C] = logRegClasif.score(X_cross_validation, y_cross_validation)

# liblinear solver
for numOfIterations in iterations:
    for C in Cs:
        logRegClasif.set_params(max_iter=numOfIterations, solver='liblinear', C=C)
        logRegClasif.fit(X_train, y_train)
        liblinearResults[numOfIterations][C] = logRegClasif.score(X_cross_validation, y_cross_validation)

# lbfgs solver
for numOfIterations in iterations:
    for C in Cs:
        logRegClasif.set_params(max_iter=numOfIterations, solver='lbfgs', C=C)
        logRegClasif.fit(X_train, y_train)
        lbfgsResults[numOfIterations][C] = logRegClasif.score(X_cross_validation, y_cross_validation)

# newton-cg solver
for numOfIterations in iterations:
    for C in Cs:
        logRegClasif.set_params(max_iter=numOfIterations, solver='newton-cg', C=C)
        logRegClasif.fit(X_train, y_train)
        newtoncgResults[numOfIterations][C] = logRegClasif.score(X_cross_validation, y_cross_validation)

# Save the data for graph making:
sagaResults.to_csv("submissions_and_results/sagaResults4c.csv")
sagResults.to_csv("submissions_and_results/sagResults4c.csv")
liblinearResults.to_csv("submissions_and_results/liblinearResults4c.csv")
lbfgsResults.to_csv("submissions_and_results/lbfgsResults4c.csv")
newtoncgResults.to_csv("submissions_and_results/newtoncgResults4c.csv")
