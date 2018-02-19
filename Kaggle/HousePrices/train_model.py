import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# load the training data
traincsv = pd.read_csv('train.csv')

# remove all features that are missing from >1% of data samples (as per eda.py output)
traincsv.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageCond', 'GarageType',
               'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond',
               'BsmtQual', 'MasVnrArea', 'MasVnrType'], axis=1)
# TODO: delete all data samples that have missing feature data (where the feature data is missing from <1% of all data samples)
indexOfNullElectricalSample = traincsv.loc[traincsv['Electrical'].isnull()].index
traincsv.drop(index=indexOfNullElectricalSample)

# get the categorical data into a numerical form (use the dummy function)
traincsv = pd.get_dummies(traincsv)

# mean normalize the numerical data
scaler = StandardScaler()
scaler.fit(traincsv)
mn_train = scaler.transform(traincsv)

# prepare linear regression model
linearRegression = LinearRegression(copy_X=True)

# TODO: train linear regression model
# TODO: test linear regression model
# TODO: prepare neural network model
# TODO: train neural network model
# TODO: test neural network model

traincsv[['GarageCond', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2',
          'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrArea', 'MasVnrType', 'Electrical']]
