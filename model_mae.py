from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
path = '../input/melbourne-housing-snapshot/melb_data.csv'
data = pd.io.parsers.read_csv(path)
predictors = ['Rooms']
y = data.Price
x = data[predictors]
train_X, val_X, train_y, val_y = train_test_split(x, y,random_state = 0)
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))    