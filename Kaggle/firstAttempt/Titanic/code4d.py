# Imports
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
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
X["PclassSquared"] = X["Pclass"] ** 2
X["SexSquared"] = (X["Sex"] + 0.5) ** 2
X["AgeSquared"] = X["Age"] ** 2
X["SibSpSquared"] = X["SibSp"] ** 2
X["ParchSquared"] = X["Parch"] ** 2
X["FareSquared"] = X["Fare"] ** 2
X["EmbarkedSquared"] = X["Embarked"] ** 2
y = training_data["Survived"]
X_train, X_cross_validation, y_train, y_cross_validation = train_test_split(X, y, test_size=(1 / 3.))

# Create classifier with parameters
logRegClasifCVSaga = LogisticRegressionCV()
logRegClasifCVSag = LogisticRegressionCV()
logRegClasifCVLiblinear = LogisticRegressionCV()
logRegClasifCVLbfgs = LogisticRegressionCV()
logRegClasifCVNewtoncg = LogisticRegressionCV()

# Create ranges of parameters
solvers = ['saga', 'sag', 'liblinear', 'lbfgs', 'newton-cg']
iterations = [100, 200, 300, 400, 500]

sagaResults = pd.Series(index=iterations)
sagResults = pd.Series(index=iterations)
liblinearResults = pd.Series(index=iterations)
lbfgsResults = pd.Series(index=iterations)
newtoncgResults = pd.Series(index=iterations)

# For each combination of parameters,
# set the classifier to have those parameters, train it, score it,
# and store the results in a dataframe

# saga solver
for numOfIterations in iterations:
    logRegClasifCVSaga.set_params(max_iter=numOfIterations, solver='saga')
    logRegClasifCVSaga.fit(X_train, y_train)
    sagaResults[numOfIterations] = logRegClasifCVSaga.score(X_cross_validation, y_cross_validation)

# sag solver
for numOfIterations in iterations:
    logRegClasifCVSag.set_params(max_iter=numOfIterations, solver='sag')
    logRegClasifCVSag.fit(X_train, y_train)
    sagResults[numOfIterations] = logRegClasifCVSag.score(X_cross_validation, y_cross_validation)

# liblinear solver
for numOfIterations in iterations:
    logRegClasifCVLiblinear.set_params(max_iter=numOfIterations, solver='liblinear')
    logRegClasifCVLiblinear.fit(X_train, y_train)
    liblinearResults[numOfIterations] = logRegClasifCVLiblinear.score(X_cross_validation, y_cross_validation)

# lbfgs solver
for numOfIterations in iterations:
    logRegClasifCVLbfgs.set_params(max_iter=numOfIterations, solver='lbfgs')
    logRegClasifCVLbfgs.fit(X_train, y_train)
    lbfgsResults[numOfIterations] = logRegClasifCVLbfgs.score(X_cross_validation, y_cross_validation)

# newton-cg solver
for numOfIterations in iterations:
    logRegClasifCVNewtoncg.set_params(max_iter=numOfIterations, solver='newton-cg')
    logRegClasifCVNewtoncg.fit(X_train, y_train)
    newtoncgResults[numOfIterations] = logRegClasifCVNewtoncg.score(X_cross_validation, y_cross_validation)

# Save the data for graph making:
sagaResults.to_csv("submissions_and_results/sagaResults4d.csv")
sagResults.to_csv("submissions_and_results/sagResults4d.csv")
liblinearResults.to_csv("submissions_and_results/liblinearResults4d.csv")
lbfgsResults.to_csv("submissions_and_results/lbfgsResults4d.csv")
newtoncgResults.to_csv("submissions_and_results/newtoncgResults4d.csv")


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
sagaSubmission.to_csv("submissions_and_results/submission4dSaga.csv", index=False)
sagSubmission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": sagPredictions})
sagSubmission.to_csv("submissions_and_results/submission4dSag.csv", index=False)
liblinearSubmission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": liblinearPredictions})
liblinearSubmission.to_csv("submissions_and_results/submission4dLiblinear.csv", index=False)
lbfgsSubmission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": lbfgsPredictions})
lbfgsSubmission.to_csv("submissions_and_results/submission4dLbfgs.csv", index=False)
newtoncgSubmission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": newtoncgPredictions})
newtoncgSubmission.to_csv("submissions_and_results/submission4dNewtoncg.csv", index=False)
