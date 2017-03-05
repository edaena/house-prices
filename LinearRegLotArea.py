import pandas as pd
from sklearn import linear_model

# Linear regression on LotArea to predict house prices, submission score on kaggle:  0.41600 (root mean squared logarithmic error)
train = pd.read_csv("train.csv", header=0, delimiter=",")

# The following are useful commands to explore the data
# print train: prints the entire dataframe
# train.head() prints the first 5 lines
# train.shape
# train MSSubClass
# train.columns.valuest

# Prepare the data with only the relevant columns
# A key component of pandas is the dataframe
reg = LinearRegression()

# Format the data for fitting the regression to avoid, if you don't do this you get an index out of range error
train_lot_area = [[x] for x in train['LotArea'].values]
reg.fit (train_lot_area , train["SalePrice")

# Read test data
test = pd.read_csv("test.csv", header=0, delimiter=",")

test_lot_area = [[x] for x in test['LotArea'].values]

# Predict and output the results to a csv file
result = reg.predict(test_lot_area)
output = pd.DataFrame(data={"Id":test["Id"], "SalePrice":result})
output.to_csv( "LinearRegLotArea.csv", index=False, quoting=3 )
