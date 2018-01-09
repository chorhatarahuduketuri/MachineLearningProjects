# Imports
import pandas as pd

# Data loading 
training_data = pd.read_csv('train.csv')

# Data preprocessing and putting into a data structure that can be 
# used by a classifier
training_data["Age"] = training_data["Age"].fillna(training_data["Age"].median())
training_data["Sex"].loc[training_data["Sex"] == "male"] = 0
training_data["Sex"].loc[training_data["Sex"] == "female"] = 1
training_data["Embarked"].fillna(0)
training_data["Embarked"].loc[training_data["Embarked"] == "C"] = 1
training_data["Embarked"].loc[training_data["Embarked"] == "Q"] = 2
training_data["Embarked"].loc[training_data["Embarked"] == "S"] = 3


# Create classifier with parameters


# Train classifier


# Load test dataset and use classifer to make predictions


# Save test predictions to disk in the format required for kaggle. 
