import pandas as pd
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
path = '../input/melbourne-housing-snapshot/melb_data.csv'
data = pd.io.parsers.read_csv(path)
predictors = ['Rooms']
y = data.Price
x = data[predictors]
model.fit(x,y)
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))
