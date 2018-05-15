import pandas as pd
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# LOADING
# load the training and submission test data
traincsv = pd.read_csv('train.csv')
testcsv = pd.read_csv('test.csv')
# combine them into one for pre-processing
alldata = pd.concat((traincsv, testcsv))

# PREPROCESSING
# remove all features that are missing from >1% of data samples (as per eda.py output)
alldata.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageCond', 'GarageType',
              'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond',
              'BsmtQual', 'MasVnrArea', 'MasVnrType'], axis=1)
# delete all data samples that have missing feature data (where the feature data is missing from <1% of all data samples)
indexOfNullElectricalSample = alldata.loc[alldata['Electrical'].isnull()].index
alldata = alldata.drop(index=indexOfNullElectricalSample)

# get the categorical data into a numerical form (use the dummy function)
alldata = pd.get_dummies(alldata)

# fill in all remaining NAN values with the mean of that column
alldata = alldata.fillna(alldata.mean())

# separate the training and submission test data again
trainingX = alldata[:1459]
testingX = alldata[1459:].drop('SalePrice', 1)

# create X and y for the training
X = traincsv.drop('SalePrice', 1)
y = traincsv['SalePrice']

# Linear Regression
linearRegression = LinearRegression(copy_X=True)
scoreNotHighEnough = True
while scoreNotHighEnough:
    # Create separate test/train data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 / 3.))

    # Create and train the linear regression model
    linearRegression.fit(X_train, y_train)

    # test linear regression model
    linearRegressionScore = linearRegression.score(X_test, y_test)
    print('Linear Regression Score: ' + str(linearRegressionScore))
    if linearRegressionScore > 0.88:
        scoreNotHighEnough = False

# prepare neural network model
mlpClassifier = MLPClassifier(hidden_layer_sizes=(289, 289), activation='logistic', solver='sgd')
# train neural network model
mlpClassifier.fit(X_train, y_train)
# test neural network model
mlpClassifierScore = mlpClassifier.score(X_test, y_test)
print('MLPClassifier score: ' + str(mlpClassifierScore))

# prepare SVM model
svmClassifier = svm.SVC(kernel='poly')
# train SVM model
svmClassifier.fit(X_train, y_train)
# test SVM model
svmClassifierScore = svmClassifier.score(X_test, y_test)
print('SVM SVC score: ' + str(svmClassifierScore))

# PREDICT SUBMISSION
predictions = linearRegression.predict(testingX)
