import pandas as pd
from sklearn import linear_model

# Read the train data
train = pd.read_csv("../train.csv", header=0, delimiter=",")

# The following are useful commands to explore the data
# print train: prints the entire dataframe
# train.head() prints the first 5 lines
# train.shape
# train MSSubClass
# train.columns.valuest

# Prediction submission scores on kaggle representing the root mean squared logarithmic error for each model trained:
# LotArea linear regression model: 0.41600
# FullBath linear regression model: 0.34693
# FirePlaces linear regression model: 0.37061
# PoolArea linear regression model: 0.42964
# YearBuilt linear regression model: 0.33743
# 1stFlrSF linear regression model: 0.33200
# 2ndFlrSF linear regression model: 0.41407
featureNames = ['LotArea', 'FullBath', 'Fireplaces', 'PoolArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF']

# Read test data
test = pd.read_csv("../test.csv", header=0, delimiter=",")

# Prepare the data with only the relevant columns
# A key component of pandas is the dataframe
reg = linear_model.LinearRegression()

for feature in featureNames:
  # Format the data for fitting the regression to avoid, if you don't do this you get an index out of range error
  train_lot_area = [[x] for x in train[feature].values]
  reg.fit (train_lot_area , train['SalePrice'])
  test_lot_area = [[x] for x in test[feature].values]

  # Predict and output the results to a csv file
  result = reg.predict(test_lot_area)
  output = pd.DataFrame(data={"Id":test["Id"], "SalePrice":result})
  fileName = "LinearReg" + feature + ".csv"
  output.to_csv( fileName, index=False, quoting=3 )
