import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

training_data = pd.read_csv('train.csv')

print('training_data.columns:')
print(training_data.columns)

print('training_data[\'SalePrice\'].describe():')
print(training_data['SalePrice'].describe())

# find out how much null data there is for features
total = training_data.isnull().sum().sort_values(ascending=False)
percent = (training_data.isnull().sum() / training_data.isnull().count()).sort_values(ascending=False)
# list the features with null data, by how much the feature is null
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print('Missing values in the training data, in absolute and proportional measurements:')
print(missing_data.head(20))

# find out about correlation among continuous-variable features
corr = training_data.corr().drop('Id', axis=0).drop('Id', axis=1)
corr_saleprice_sorted = corr['SalePrice'].sort_values(ascending=False)
print('Correlation of continuous-value features with SalePrice: ')
print(corr_saleprice_sorted)

# make some graphs showing the correlation between sale price and the most highly correlated features (everything >= 0.5)
x = training_data['SalePrice']
y = training_data['OverallQual']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('OverallQual')
plt.title('SalePrice vs OverallQual')
plt.savefig('gitIgnoreDir/scatterSalePriceVsOverallQual.png')
plt.clf()

x = training_data['SalePrice']
y = training_data['GrLivArea']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('GrLivArea')
plt.title('SalePrice vs GrLivArea')
plt.savefig('gitIgnoreDir/scatterSalePriceVsGrLivArea.png')
plt.clf()

x = training_data['SalePrice']
y = training_data['GarageCars']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('GarageCars')
plt.title('SalePrice vs GarageCars')
plt.savefig('gitIgnoreDir/scatterSalePriceVsGarageCars.png')
plt.clf()

x = training_data['SalePrice']
y = training_data['GarageArea']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('GarageArea')
plt.title('SalePrice vs GarageArea')
plt.savefig('gitIgnoreDir/scatterSalePriceVsGarageArea.png')
plt.clf()

x = training_data['SalePrice']
y = training_data['TotalBsmtSF']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('TotalBsmtSF')
plt.title('SalePrice vs TotalBsmtSF')
plt.savefig('gitIgnoreDir/scatterSalePriceVsTotalBsmtSF.png')
plt.clf()

x = training_data['SalePrice']
y = training_data['1stFlrSF']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('1stFlrSF')
plt.title('SalePrice vs 1stFlrSF')
plt.savefig('gitIgnoreDir/scatterSalePriceVs1stFlrSF.png')
plt.clf()

x = training_data['SalePrice']
y = training_data['FullBath']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('FullBath')
plt.title('SalePrice vs FullBath')
plt.savefig('gitIgnoreDir/scatterSalePriceVsFullBath.png')
plt.clf()

x = training_data['SalePrice']
y = training_data['TotRmsAbvGrd']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('TotRmsAbvGrd')
plt.title('SalePrice vs TotRmsAbvGrd')
plt.savefig('gitIgnoreDir/scatterSalePriceVsTotRmsAbvGrd.png')
plt.clf()

x = training_data['SalePrice']
y = training_data['YearBuilt']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('YearBuilt')
plt.title('SalePrice vs YearBuilt')
plt.savefig('gitIgnoreDir/scatterSalePriceVsYearBuilt.png')
plt.clf()

x = training_data['SalePrice']
y = training_data['YearRemodAdd']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('YearRemodAdd')
plt.title('SalePrice vs YearRemodAdd')
plt.savefig('gitIgnoreDir/scatterSalePriceVsYearRemodAdd.png')
plt.clf()

# TODO: make graphs of correlation between continuous-variable features
plt.clf()
sns.set()
cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
        '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
sns.pairplot(training_data[cols], size=2.5)
plt.savefig('gitIgnoreDir/topTenCorrPairPlot.png')

# find out what features correlate with each other very highly 

# TODO: use this for anything that seemed interesting
#sns.jointplot()