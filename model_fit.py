import pandas as pd
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
path = '../input/melbourne-housing-snapshot/melb_data.csv'
data = pd.io.parsers.read_csv(path)
predictors = ['Rooms']
y = data.Price
x = data[predictors]
model.fit(x,y)
