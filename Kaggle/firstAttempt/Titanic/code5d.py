# Imports
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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

pca = PCA()
lr = LogisticRegression(max_iter=500, solver='lbfgs')

# Run the training 1000 times, store results in results
pca.set_params(n_components=7)
pca.fit(X, y)
X_PCA = pca.fit_transform(X)
lr.fit(X_PCA, y)

# Load submission data and prepare for submission
submission_data = pd.read_csv('test.csv')
submission_data["Age"] = submission_data["Age"].fillna(submission_data["Age"].median())
submission_data["Sex"].loc[submission_data["Sex"] == "male"] = 0
submission_data["Sex"].loc[submission_data["Sex"] == "female"] = 1
submission_data["Embarked"] = submission_data["Embarked"].fillna(0)
submission_data["Embarked"].loc[submission_data["Embarked"] == "C"] = 1
submission_data["Embarked"].loc[submission_data["Embarked"] == "Q"] = 2
submission_data["Embarked"].loc[submission_data["Embarked"] == "S"] = 3
submission_data["Fare"] = submission_data["Fare"].fillna(submission_data["Fare"].median())
# Form submission data structure containing the relevant data
X_submission = submission_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

# Fit PCA transform to submission data and predict survival
X_submission_PCA = pca.fit_transform(X_submission)
predictions = lr.predict(X_submission_PCA)
# Save to disk in appropriate format
submission = pd.DataFrame({"PassengerId": submission_data["PassengerId"], "Survived": predictions})
submission.to_csv("submissions_and_results/submission5d.csv", index=False)
