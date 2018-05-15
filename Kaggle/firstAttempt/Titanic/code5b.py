# Imports
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

# Use PCA to create different data structures to feed into logistic regression
pca_n_components_1 = PCA(n_components=1)
pca_n_components_2 = PCA(n_components=2)
pca_n_components_3 = PCA(n_components=3)
pca_n_components_4 = PCA(n_components=4)
pca_n_components_5 = PCA(n_components=5)
pca_n_components_6 = PCA(n_components=6)
pca_n_components_7 = PCA(n_components=7)

pca_n_components_1.fit(X)
pca_n_components_2.fit(X)
pca_n_components_3.fit(X)
pca_n_components_4.fit(X)
pca_n_components_5.fit(X)
pca_n_components_6.fit(X)
pca_n_components_7.fit(X)

X_PCA_1 = pca_n_components_1.fit_transform(X)
X_PCA_2 = pca_n_components_2.fit_transform(X)
X_PCA_3 = pca_n_components_3.fit_transform(X)
X_PCA_4 = pca_n_components_4.fit_transform(X)
X_PCA_5 = pca_n_components_5.fit_transform(X)
X_PCA_6 = pca_n_components_6.fit_transform(X)
X_PCA_7 = pca_n_components_7.fit_transform(X)

# Split the PCA'd training data into training and test sets,
# so that the logistic regression models can be tested.
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_PCA_1, y, test_size=(1 / 5.))
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_PCA_2, y, test_size=(1 / 5.))
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_PCA_3, y, test_size=(1 / 5.))
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_PCA_4, y, test_size=(1 / 5.))
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_PCA_5, y, test_size=(1 / 5.))
X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(X_PCA_6, y, test_size=(1 / 5.))
X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(X_PCA_7, y, test_size=(1 / 5.))

# Create classifier with parameters
logisticRegressionClassifier_1 = LogisticRegression(max_iter=500, solver='lbfgs')
logisticRegressionClassifier_2 = LogisticRegression(max_iter=500, solver='lbfgs')
logisticRegressionClassifier_3 = LogisticRegression(max_iter=500, solver='lbfgs')
logisticRegressionClassifier_4 = LogisticRegression(max_iter=500, solver='lbfgs')
logisticRegressionClassifier_5 = LogisticRegression(max_iter=500, solver='lbfgs')
logisticRegressionClassifier_6 = LogisticRegression(max_iter=500, solver='lbfgs')
logisticRegressionClassifier_7 = LogisticRegression(max_iter=500, solver='lbfgs')

# Train classifier
logisticRegressionClassifier_1.fit(X_train_1, y_train_1)
logisticRegressionClassifier_2.fit(X_train_2, y_train_2)
logisticRegressionClassifier_3.fit(X_train_3, y_train_3)
logisticRegressionClassifier_4.fit(X_train_4, y_train_4)
logisticRegressionClassifier_5.fit(X_train_5, y_train_5)
logisticRegressionClassifier_6.fit(X_train_6, y_train_6)
logisticRegressionClassifier_7.fit(X_train_7, y_train_7)

print(
    """PCA_n_components_1 test set score: """ + str(logisticRegressionClassifier_1.score(X_test_1, y_test_1)) + '\n' +
    """PCA_n_components_2 test set score: """ + str(logisticRegressionClassifier_2.score(X_test_2, y_test_2)) + '\n' +
    """PCA_n_components_3 test set score: """ + str(logisticRegressionClassifier_3.score(X_test_3, y_test_3)) + '\n' +
    """PCA_n_components_4 test set score: """ + str(logisticRegressionClassifier_4.score(X_test_4, y_test_4)) + '\n' +
    """PCA_n_components_5 test set score: """ + str(logisticRegressionClassifier_5.score(X_test_5, y_test_5)) + '\n' +
    """PCA_n_components_6 test set score: """ + str(logisticRegressionClassifier_6.score(X_test_6, y_test_6)) + '\n' +
    """PCA_n_components_7 test set score: """ + str(logisticRegressionClassifier_7.score(X_test_7, y_test_7)))
