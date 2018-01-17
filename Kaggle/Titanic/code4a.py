# Imports
import pandas as pd
import numpy as np
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
y = training_data["Survived"]
X_train, X_cross_validation, y_train, y_cross_validation = train_test_split(X, y, test_size=(1/3.))

# Create classifier with parameters
logRegClasif = LogisticRegression()

# Create ranges of parameters
solvers = ['saga', 'sag', 'liblinear', 'lbfgs', 'newton-cg']
iterations = [100,200,300,400,500]
Cs = np.linspace(1e-4,1e4)

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
        sagaResults[numOfIterations][C] = logRegClasif.score(X_cross_validation,y_cross_validation)

# sag solver
for numOfIterations in iterations:
    for C in Cs:
        logRegClasif.set_params(max_iter=numOfIterations, solver='sag', C=C)
        logRegClasif.fit(X_train, y_train)
        sagResults[numOfIterations][C] = logRegClasif.score(X_cross_validation,y_cross_validation)

# liblinear solver
for numOfIterations in iterations:
    for C in Cs:
        logRegClasif.set_params(max_iter=numOfIterations, solver='liblinear', C=C)
        logRegClasif.fit(X_train, y_train)
        liblinearResults[numOfIterations][C] = logRegClasif.score(X_cross_validation,y_cross_validation)

# lbfgs solver
for numOfIterations in iterations:
    for C in Cs:
        logRegClasif.set_params(max_iter=numOfIterations, solver='lbfgs', C=C)
        logRegClasif.fit(X_train, y_train)
        lbfgsResults[numOfIterations][C] = logRegClasif.score(X_cross_validation,y_cross_validation)

# newton-cg solver
for numOfIterations in iterations:
    for C in Cs:
        logRegClasif.set_params(max_iter=numOfIterations, solver='newton-cg', C=C)
        logRegClasif.fit(X_train, y_train)
        newtoncgResults[numOfIterations][C] = logRegClasif.score(X_cross_validation,y_cross_validation)

# Save the data for graph making:
sagaResults.to_csv("sagaResults.csv")
sagResults.to_csv("sagResults.csv")
liblinearResults.to_csv("liblinearResults.csv")
lbfgsResults.to_csv("lbfgsResults.csv")
newtoncgResults.to_csv("newtoncgResults.csv")

classifier = LogisticRegression()
classifier.fit(X_train,y_train)
score = classifier.score(X_cross_validation,y_cross_validation)
print("Score of default LogisticRegression classifier object: %n", (score)) # 0.787878787879

# # Load test dataset and use classifier to make predictions
# test_data = pd.read_csv('test.csv')
#
# test_data["Age"] = test_data["Age"].fillna(test_data["Age"].median())
# test_data["Sex"].loc[test_data["Sex"] == "male"] = 0
# test_data["Sex"].loc[test_data["Sex"] == "female"] = 1
# test_data["Embarked"] = test_data["Embarked"].fillna(0)
# test_data["Embarked"].loc[test_data["Embarked"] == "C"] = 1
# test_data["Embarked"].loc[test_data["Embarked"] == "Q"] = 2
# test_data["Embarked"].loc[test_data["Embarked"] == "S"] = 3
# test_data["Fare"] = test_data["Fare"].fillna(0)
#
# X_test = test_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
#
# predictions = logRegClasif.predict(X_test)
#
# # Save test predictions to disk in the format required for kaggle.
# submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": predictions})
# submission.to_csv("submission4.csv", index=False)
