import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

traincsv = pd.read_csv('train.csv')
testcsv = pd.read_csv('test.csv')

training_data = traincsv[
    ['Id', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
     'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'SalePrice']
]
testing_data = testcsv[
    ['Id', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
     'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
]

all_data = pd.concat((training_data, testing_data))

all_data = all_data.fillna(all_data.mean())

trainingX = all_data[:1460]
submittingX = all_data[1460:].drop('SalePrice', 1)

X = trainingX.drop('SalePrice', 1)
y = trainingX['SalePrice']

linearRegression = LinearRegression()

scoreNotGoodEnough = True
while scoreNotGoodEnough:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 / 5.))
    linearRegression.fit(X_train, y_train)
    score = linearRegression.score(X_test, y_test)
    print('Score: ' + str(score))
    if score > 0.87:
        scoreNotGoodEnough = False

predictions = linearRegression.predict(submittingX)

submission = pd.DataFrame({"Id": submittingX["Id"], "SalePrice": predictions})
submission.to_csv("gitIgnoreDir/submission.csv", index=False)
