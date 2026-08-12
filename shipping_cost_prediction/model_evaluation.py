from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_linear_regression(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def train_decision_tree(X_train, Y_train ,max_depth=5):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, Y_train)
    return model 


def train_random_forest(X_train, Y_train, max_depth=5):
    model = RandomForestRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, Y_train)
    return model 


def evaluate_model(model, X_test, Y_test, model_name: str) -> dict:
    '''
    evaluate regression model and return metrics.
    '''
    prediction = model.predict(X_test)

    mae = mean_absolute_error(prediction,Y_test)
    rmse = mean_squared_error(prediction,Y_test)
    r2 = r2_score(prediction,Y_test)*100

    return {
        "model_name": model_name,
        "mae":  mae,
        "rmse":  rmse,
        "r2":  r2
    }
    