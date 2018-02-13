import matplotlib.pyplot as plt
import pandas as pd

training_data = pd.read_csv('train.csv')

print('training_data.columns:')
training_data.columns

print('training_data[\'SalePrice\'].describe():')
training_data['SalePrice'].describe()

# find out how much null data there is for features
total = training_data.isnull().sum().sort_values(ascending=False)
percent = (training_data.isnull().sum() / training_data.isnull().count()).sort_values(ascending=False)
# list the features with null data, by how much the feature is null
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print('Missing values in the training data, in absolute and proportional measurements:')
missing_data.head(20)

# find out about correlation among continuous-variable features
corr = training_data.corr().drop('Id', axis=0).drop('Id', axis=1)
corr_saleprice_sorted = corr['SalePrice'].sort_values(ascending=False)
print('Correlation of continuous-value features with SalePrice: ')
corr_saleprice_sorted

# TODO: make some graphs showing the correlation between sale price and the most highly correlated features
# TODO: (everything >= 0.5)
x = training_data['SalePrice']
y = training_data['OverallQual']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('OverallQual')
plt.title('SalePrice vs OverallQual')
plt.show()

x = training_data['SalePrice']
y = training_data['GrLivArea']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('GrLivArea')
plt.title('SalePrice vs GrLivArea')
plt.show()

x = training_data['SalePrice']
y = training_data['GarageCars']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('GarageCars')
plt.title('SalePrice vs GarageCars')
plt.show()

x = training_data['SalePrice']
y = training_data['GarageArea']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('GarageArea')
plt.title('SalePrice vs GarageArea')
plt.show()

x = training_data['SalePrice']
y = training_data['TotalBsmtSF']
plt.scatter(x, y, marker='x')
plt.xlabel('SalePrice')
plt.ylabel('TotalBsmtSF')
plt.title('SalePrice vs TotalBsmtSF')
plt.show()


# TODO: make graphs of correlation between continuous-variable features
