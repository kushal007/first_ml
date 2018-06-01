import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
path = '../input/melbourne-housing-snapshot/melb_data.csv'
data = pd.io.parsers.read_csv(path)
predictors = ['Rooms']
y = data.Price
x = data[predictors]
train_X, val_X, train_y, val_y = train_test_split(x, y,random_state = 0)
model = RandomForestRegressor()
model.fit(train_X, train_y)
model_prediction = model.predict(val_X)
print(mean_absolute_error(val_y, model_prediction))
