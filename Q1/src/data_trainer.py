from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import data_preprocessing  
import joblib 

# Loading the sacaled dataset from preprocessig file.
X,y = data_preprocessing.load_transform_dataset()
# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 , random_state = 42 )

X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)

# Create a variable to store the path of the pkl files 
modelfolder = 'C:/Users/desut/Desktop/Shannu_ML/Q1/src/models'

# Simple linear regression model.
def Simple_Linear_Regression():
    # Considering ZN feature to train test
    X_slr = X_train[['ZN']]
    x_test_slr = X_test[['ZN']]
# Fitting the model with train data
    slr_model = LinearRegression()
    slr_model.fit(X_slr, y_train)
# dumping the pkl files 
    joblib.dump(slr_model, modelfolder+'/slr_model.pkl')
    joblib.dump((x_test_slr, y_test), modelfolder+'/slr_model_test_data.pkl') 


# Mulitple Linear Regression model considering all the features .
def Multi_Linear_Regression():
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    joblib.dump(lr_model, modelfolder+'/lr_model.pkl')
    joblib.dump((X_test, y_test), modelfolder+'/lr_model_test_data.pkl') 


# ridge, lasso and ols
# Adding alpha- penalty to the model
def Ridge_regression_model(alpha=1.0):
    ridge_model = Ridge(alpha = alpha)
    ridge_model.fit(X_train, y_train)

    joblib.dump(ridge_model, modelfolder+'/ridge_model.pkl')
    joblib.dump((X_test, y_test), modelfolder+'/ridge_model_test_data.pkl') 

# Shrink the coefficents to zero.
def Lasso_regression_model(alpha=1.0):
    lasso_model = Lasso(alpha = alpha)
    lasso_model.fit(X_train, y_train)

    joblib.dump(lasso_model, modelfolder+'/lasso_model.pkl')
    joblib.dump((X_test, y_test), modelfolder+'/lasso_model_test_data.pkl') 

# OLS to identify the important features.
def OLS_Regression():
    # Adding constant for OLS regression (intercept is already present)
    X_train_ols = sm.add_constant(X_train, prepend=False)
    ols_model = sm.OLS(y_train, X_train_ols).fit()

    # Saving the model
    joblib.dump(ols_model, modelfolder+'/ols_model.pkl')
    joblib.dump((X_test, y_test), modelfolder+'/ols_model_test_data.pkl') 

    return ols_model









    
if __name__ == "__main__":

# Started running OLS model and printing the  summary
    print("Running OLS Regression")
    ols_model = OLS_Regression()
    print(ols_model.summary())

# Started runnig Simple linear regression
    print("Running Simple Linear Regression")
    Simple_Linear_Regression()

# Started running mutiple linear regression
    print('Running Multi Linear Regression')
    Multi_Linear_Regression()

# Running ridge model with default alpha=1.0
    # print("Running Ridge Model")
    # Ridge_regression_model()

# Tweaking the model with different alpha values for better performance
    print("Running Ridge Model")
    Ridge_regression_model(alpha=0.1)

    # Tweaking the model with different alpha values for better performance
    print("Running Lasso Model")
    Lasso_regression_model(alpha=0.1)

  