import joblib
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

modelfolder = 'C:/Users/desut/Desktop/Shannu_ML/Q1/src/models'

# Method to load the pkl files
def model_data(Ml_model_name):
    mlmodel = joblib.load(f'{modelfolder}/{Ml_model_name}_model.pkl')
    X_test, y_test = joblib.load(f'{modelfolder}/{Ml_model_name}_model_test_data.pkl')
    return mlmodel, X_test, y_test

# function to evaluate the metrics of each model.

def evaluate_model_metrics(mlmodel, X_test, y_test, is_OLS = False):
    # if it is Ols model we need to add constant before predicting. 
    if is_OLS:
        X_test = sm.add_constant(X_test, prepend = False)
    y_pred = mlmodel.predict(X_test)

# for the remaining we can directly get the metrics.
    mserros = mean_squared_error(y_test, y_pred)
    r2_scores = r2_score(y_test, y_pred)

    return mserros, r2_scores

# Creating a disctionary for models 
models = {

    'slr': 'Simple Linear Regression',
    'lr':'Linear Regression',
    'ridge':' Ridge Regression',
    'lasso':'Lasso Regression',
    'ols': 'OLS Regression'
}

# Displaying the metrics of each model.
for mod_name, mod_desc in models.items():
    mlmodel, X_test, y_test = model_data(mod_name)
    is_OLS = (mod_name == 'ols')
    mse,r2 = evaluate_model_metrics(mlmodel, X_test, y_test, is_OLS=is_OLS)

    print(f'{mod_desc} - MSE: {mse : .4f}, R-squared: {r2:4f}')



