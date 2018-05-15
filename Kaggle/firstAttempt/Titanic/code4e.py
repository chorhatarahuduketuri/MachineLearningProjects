# Imports
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

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

# Generate training set
X = training_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
X["PclassSquared"] = X["Pclass"] ** 2
X["SexSquared"] = (X["Sex"] + 0.5) ** 2
X["AgeSquared"] = X["Age"] ** 2
X["SibSpSquared"] = X["SibSp"] ** 2
X["ParchSquared"] = X["Parch"] ** 2
X["FareSquared"] = X["Fare"] ** 2
X["EmbarkedSquared"] = X["Embarked"] ** 2
y = training_data["Survived"]

# Create classifiers
logRegClasifCVSaga = LogisticRegressionCV()
logRegClasifCVSag = LogisticRegressionCV()
logRegClasifCVLiblinear = LogisticRegressionCV()
logRegClasifCVLbfgs = LogisticRegressionCV()
logRegClasifCVNewtoncg = LogisticRegressionCV()

# For each solver: set params, train it:
# saga solver
logRegClasifCVSaga.set_params(max_iter=500, solver='saga')
logRegClasifCVSaga.fit(X, y)

# sag solver
logRegClasifCVSag.set_params(max_iter=500, solver='sag')
logRegClasifCVSag.fit(X, y)

# liblinear solver
logRegClasifCVLiblinear.set_params(max_iter=500, solver='liblinear')
logRegClasifCVLiblinear.fit(X, y)

# lbfgs solver
logRegClasifCVLbfgs.set_params(max_iter=500, solver='lbfgs')
logRegClasifCVLbfgs.fit(X, y)

# newton-cg solver
logRegClasifCVNewtoncg.set_params(max_iter=500, solver='newton-cg')
logRegClasifCVNewtoncg.fit(X, y)

# Load test dataset and use classifier to make predictions
test_data = pd.read_csv('test.csv')

test_data["Age"] = test_data["Age"].fillna(test_data["Age"].median())
test_data["Sex"].loc[test_data["Sex"] == "male"] = 0
test_data["Sex"].loc[test_data["Sex"] == "female"] = 1
test_data["Embarked"] = test_data["Embarked"].fillna(0)
test_data["Embarked"].loc[test_data["Embarked"] == "C"] = 1
test_data["Embarked"].loc[test_data["Embarked"] == "Q"] = 2
test_data["Embarked"].loc[test_data["Embarked"] == "S"] = 3
test_data["Fare"] = test_data["Fare"].fillna(0)

X_test = test_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
X_test["PclassSquared"] = X["Pclass"] ** 2
X_test["SexSquared"] = (X["Sex"] + 0.5) ** 2
X_test["AgeSquared"] = X["Age"] ** 2
X_test["SibSpSquared"] = X["SibSp"] ** 2
X_test["ParchSquared"] = X["Parch"] ** 2
X_test["FareSquared"] = X["Fare"] ** 2
X_test["EmbarkedSquared"] = X["Embarked"] ** 2

sagaPredictions = logRegClasifCVSaga.predict(X_test)
sagPredictions = logRegClasifCVSag.predict(X_test)
liblinearPredictions = logRegClasifCVLiblinear.predict(X_test)
lbfgsPredictions = logRegClasifCVLbfgs.predict(X_test)
newtoncgPredictions = logRegClasifCVNewtoncg.predict(X_test)

# Save test predictions to disk in the format required for kaggle.
sagaSubmission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": sagaPredictions})
sagaSubmission.to_csv("submissions_and_results/submission4eSaga.csv", index=False)
sagSubmission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": sagPredictions})
sagSubmission.to_csv("submissions_and_results/submission4eSag.csv", index=False)
liblinearSubmission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": liblinearPredictions})
liblinearSubmission.to_csv("submissions_and_results/submission4eLiblinear.csv", index=False)
lbfgsSubmission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": lbfgsPredictions})
lbfgsSubmission.to_csv("submissions_and_results/submission4eLbfgs.csv", index=False)
newtoncgSubmission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": newtoncgPredictions})
newtoncgSubmission.to_csv("submissions_and_results/submission4eNewtoncg.csv", index=False)
