# house-prices
Predict house prices

List of variables in the training data:

'Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
       'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir',
       'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
       'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
       'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold',
       'YrSold', 'SaleType', 'SaleCondition', 'SalePrice'], dtype=object
       
To-do:
1. Predict SalePrice
2. Determine what variables to use
3. Try different machine learning algorithms

## Linear Regression
The training and testing data contains information that may or may not be useful for predicting the price. Looking at the data, 'LotArea', 'FullBath', 'Fireplaces', 'PoolArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF' are attributes of a house that can be related to the cost. 

Coming from north Mexico, PoolArea seemed to me to be the best candidate =D.

To run the models:
```
python LinearRegression.py
```

### Results
A total of 7 linear regression models were trained. One for each feature mentioned above. The models were submitted on Kaggle. The table below shows the value for the squared logarithmic error for each model.

Feature | squared logarithmic error
------------ | -------------
1stFlrSF | 0.33200
YearBuilt | 0.33743
FullBath | 0.34693
FirePlaces | 0.37061
2ndFlrSF | 0.41407
LotArea | 0.41600
PoolArea | 0.42964

The smallest squared logarithmic error was from the 1stFlrSF model which represents the square feet of the 1st floor.
