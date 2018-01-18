# Imports
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Data loading
training_data = pd.read_csv('train.csv')

# Data preprocessing and putting into a data structure that can be
# used by a classifier
training_data["Age"] = training_data["Age"].fillna(0)
training_data["Sex"].loc[training_data["Sex"] == "male"] = 0
training_data["Sex"].loc[training_data["Sex"] == "female"] = 1
training_data["Embarked"] = training_data["Embarked"].fillna(0)
training_data["Embarked"].loc[training_data["Embarked"] == "C"] = 1
training_data["Embarked"].loc[training_data["Embarked"] == "Q"] = 2
training_data["Embarked"].loc[training_data["Embarked"] == "S"] = 3

X = training_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = training_data["Survived"]

# Create classifier with parameters
logRegClasif = LogisticRegression(max_iter=500, solver='lbfgs')

# Train classifier
logRegClasif.fit(X, y)

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

predictions = logRegClasif.predict(X_test)

# Save test predictions to disk in the format required for kaggle.
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": predictions})
submission.to_csv("submissions_and_results/submission3.csv", index=False)
