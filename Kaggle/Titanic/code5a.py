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

# Create classifier with parameters
logisticRegressionClassifier_1 = LogisticRegression(max_iter=500, solver='lbfgs')
logisticRegressionClassifier_2 = LogisticRegression(max_iter=500, solver='lbfgs')
logisticRegressionClassifier_3 = LogisticRegression(max_iter=500, solver='lbfgs')
logisticRegressionClassifier_4 = LogisticRegression(max_iter=500, solver='lbfgs')
logisticRegressionClassifier_5 = LogisticRegression(max_iter=500, solver='lbfgs')
logisticRegressionClassifier_6 = LogisticRegression(max_iter=500, solver='lbfgs')
logisticRegressionClassifier_7 = LogisticRegression(max_iter=500, solver='lbfgs')

# Train classifier
logisticRegressionClassifier_1.fit(X_PCA_1, y)
logisticRegressionClassifier_2.fit(X_PCA_2, y)
logisticRegressionClassifier_3.fit(X_PCA_3, y)
logisticRegressionClassifier_4.fit(X_PCA_4, y)
logisticRegressionClassifier_5.fit(X_PCA_5, y)
logisticRegressionClassifier_6.fit(X_PCA_6, y)
logisticRegressionClassifier_7.fit(X_PCA_7, y)

print(
    """n_components_1 score: """ + str(logisticRegressionClassifier_1.score(X_PCA_1, y)) + '\n' +
    """n_components_2 score: """ + str(logisticRegressionClassifier_2.score(X_PCA_2, y)) + '\n' +
    """n_components_3 score: """ + str(logisticRegressionClassifier_3.score(X_PCA_3, y)) + '\n' +
    """n_components_4 score: """ + str(logisticRegressionClassifier_4.score(X_PCA_4, y)) + '\n' +
    """n_components_5 score: """ + str(logisticRegressionClassifier_5.score(X_PCA_5, y)) + '\n' +
    """n_components_6 score: """ + str(logisticRegressionClassifier_6.score(X_PCA_6, y)) + '\n' +
    """n_components_7 score: """ + str(logisticRegressionClassifier_7.score(X_PCA_7, y)))
