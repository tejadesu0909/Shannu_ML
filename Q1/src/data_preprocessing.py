# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# To open the plotly visualz in a browser. 
pio.renderers.default = 'browser'

# Created seperated requirements.txt file to download all the libraries in a single file.

# This method will helps us to load, clean, pre process, transform, feature scaling the dataset and returns the scaled dataset.
def load_transform_dataset():
    # Loading the dataset.
    housing = pd.read_csv('C:\\Users\\desut\\Desktop\\Shannu_ML\\Q1\\HousingData.csv')
    # Creating another copy of the original data set.
    df = housing.copy()
    # To see No.of Rows and columns of the dataset.
    print(f'No.of Rows and columns:',df.shape)
    # To see what are the columns in the dataset.
    print(f'Columns in the dataset are:',df.columns)
    # Check for null values and display the count for each column.
    print(df.isnull().sum())
    # There are null values for couple of columns, need to check how much percentage there are? If we have less percentage of null values we can drop them.
    print(f'The percentage of null values for each column: before dropping them:',df.isnull().mean()*100)
    # there are 6 columns with not more than 4%, hence dropig all the null values.
    df = df.dropna()
    # Checking the percentage of null values, after dropping them.
    print(f'The percentage of null values for each column: before dropping them:', df.isnull().mean()*100)
    # To check for duplicated rows.
    print(f"no.of duplicated rows =",df.duplicated().sum())
    # To see the data types of each feature.
    print(df.info())

    # Data types are also in correct format only!

    # Data Visualization 
    # Plotting Histogram using Seaborn to understand the distribution of each feature and gain some insights much about the dataset.
    fig, ax = plt.subplots(2,7, figsize = (20,10))
    ax = ax.flatten()
    for index, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, ax=ax[index])
        ax[index].set_title(f"distribution of {col}")
        ax[index].set_xlabel(col)
        ax[index].set_ylabel('Frequency')
    plt.tight_layout()
    # Saving the figures into their respective folder
    plt.savefig('Q1/src/models/Visualizations/histplot.png')
    plt.show()
    '''
        Using the above dist plot we observed: 
        1. there is a right skewness for several features like CRIM, ZN, LSTAT, DIS, and MEDV
        2. As they are right skewed, the valeus are nearer to 0, which tells us that:
            a. CRIM is right skewed means, most of the houses in the dataset has low crime rates.
            b. NOX is also right skewed means most of the houses are with high concentration of Nitric oxide.
            c. DIS is also right skewed means most of the most areas are closer to the employment careers.
            d. PTRATIO is also right skwewd means most of the areas having moderate people teacher ratio.
            e. LSTAT is rightskwed means most of the houses are having a lower percentage of lower status individuals. 
        3. Couple of columns are BiModal such as INDUS, TAX which says:
            a. Industrial activity with low and high.
            b. tax brackets or regions with low and high. 

    '''
# Plotting the regplot to identify some more insghts about the data.
    fig, ax = plt.subplots(2,7, figsize = (20,10))
    ax = ax.flatten()
    for index, col in enumerate(df.columns):
        # sns.scatterplot(x = df[col], y = df['MEDV'], ax=ax[index], markers='X')
        sns.regplot(x = df[col], y = df['MEDV'], ax=ax[index])
        ax[index].set_title(f"{col} vs MEDV")
        ax[index].set_xlabel(col)
        ax[index].set_ylabel('MEDV')
        
    plt.tight_layout()
    # Saving the figures into their respective folder
    plt.savefig('Q1/src/models/Visualizations/regplot.png')
    plt.show()

    '''
        With the help of this scatter plots against MEDV: 
            1. Negative relation between crime rate and house prices, higher crime rate- lower house price.
            2. slightly negative relation between indus and medv, higher industrial activity- lower house price. 
            3. houses near to the charles river are slightly higher compared to the houses far from river. 
            4. Many points are areound AGe of 100, which means many houses are older than 100 years. 
            5. positive relation between dis and medv, housed near to th eemployment centre are cheaper.

    '''

    # Plotting Heat map to check for multicollinearity  within the feature variables.
    sns.heatmap(df.corr(), annot = True, fmt=".1f", linewidths=0.5, cmap= sns.light_palette("seagreen", as_cmap=True))
    plt.title('Heatmap')
    # Saving the figures into their respective folder
    plt.savefig('Q1/src/models/Visualizations/heatmap.png')
    plt.show()

    # A VIF of 1 indicates that the feature has no correlation with any of the other features.

    # Drop the target variable from the features
    X = df.drop(columns='MEDV')
    # make sure it is pandas data frame.
    X = pd.DataFrame(X)
    # Adding intercept to calcluate the VIF which helps us to address multicollinearity
    X.insert(0, 'Intercept', 1)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_df = pd.DataFrame()
    vif_df['features'] = X.columns
    vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1]) ]
    print(vif_df.sort_values(by='VIF', ascending=False))

    '''
    After calculating VIF values we observed that tax has more multicollinearity than RAD, hence dropping TAX as well from features.

    '''


    X = X.drop(columns='TAX')
    print(X.columns)

    # Now,we will check for outliers, 
    # fig, ax = plt.subplots(2,6, figsize=(20,10))
    # ax = ax.flatten()
    # for index,col in enumerate(X.columns[1:]):
    #     sns.boxplot(x= X[col], ax = ax[index])
    #     ax[index].set_title(f'Boxplot of {col}')
    # plt.tight_layout()
    # plt.show()

    # We are using plotly library for boxplots because, it visually displays the upper and lower fencesvalues. which will helps us to clip those values.

    fig = make_subplots(rows=2, cols=6, subplot_titles=X.columns[1:])
    for index, col in enumerate(X.columns[1:]):
        row = (index//6)+1
        col_pos = (index%6)+1

        fig.add_trace(
            go.Box(y=X[col], name=col),
            row = row, col = col_pos)
    fig.update_layout(height = 800, width = 1400, title_text = 'Boxplot of features with outliers')
    fig.show()

    # Features with visible outliers and their fences values
    fences = {
        'CRIM': {'lower':0.00632, 'upper':8.26725},
        'ZN': {'lower':0, 'upper':30},
        'RM': {'lower':4.903, 'upper':7.691},
        'DIS': {'lower':1.1296, 'upper':9.2229},
        'B': {'lower':347.88, 'upper':396.9},
        'LSTAT': {'lower':1.73, 'upper':31.99}
    }
# using pandas clip function capping the values.
    for col, fence in fences.items():
        X[col] = X[col].clip(lower=fence['lower'], upper=fence['upper'])
# Make sure that there are no outliers after capping.
    fig = make_subplots(rows=2, cols=6, subplot_titles=X.columns[1:])
    for index, col in enumerate(X.columns[1:]):
        row = (index//6)+1
        col_pos = (index%6)+1
        fig.add_trace(
            go.Box(y=X[col], name=col),
            row = row, col = col_pos)
    fig.update_layout(height = 800, width = 1400, title_text = 'Boxplot of features after removing outliers')
    fig.show()

# When we plot histplot, we observed some of the features are right skewed, to handle skewness, we will apply transformation of Features. 
    skewness = X.skew()
    print('Skewness of each feature before doing transformations:\n', skewness.sort_values(ascending= False))
    # If the skewness value is greater than 1 they are right skewed, 
    # If the skewness values is less than 1 they are left skewed.

    '''
    Skewness of each feature:
    CHAS         3.428643
    CRIM         1.268575
    ZN           1.267523
    RAD          1.050144
    DIS          0.875539
    LSTAT        0.831029
    NOX          0.703377
    INDUS        0.358792
    RM           0.356667
    Intercept    0.000000
    AGE         -0.594880
    PTRATIO     -0.884475
    B           -1.168637
    '''
# seeing the data fram before skewness.
    print(f'Before skewness:/n',X.sample(5))
    # Handling Left skewed Features
    X['AGE'] = X['AGE']**3
    X['PTRATIO'] = X['PTRATIO']**3
    X['B'] = X['AGE']**2

    # Handling Right Skewed Features
    X['CRIM'] = np.log1p(X['CRIM'])
    X['ZN'] = np.log1p(X['ZN'])
    X['RAD'] = np.log1p(X['RAD'])

# checking the skewness after tranformations, 
    skewness_after = X.skew()
    print('Skewness of each feature After doing transformations:\n', skewness_after.sort_values(ascending= False))
    print(f'after skewness',X.sample(5))

# Target Variable
    y = df['MEDV']

    # Feature scaling using MinMax scaler (while doing the featurescaling , we are not including the interecept column, any ways it will be same value of 1).
    # scaler = StandardScaler()
    # Standard scaler is not working properly for this dataset.
    scaler = MinMaxScaler()
    X_features = X.iloc[:, 1:]
    X_scaled_features = scaler.fit_transform(X_features)
    X_scaled = pd.DataFrame(X_scaled_features, columns=X.columns[1:])
    X_scaled.insert(0,'Intercept', X['Intercept'].values)
    print(f'After standardization',X_scaled.sample(5))
    return X_scaled,y








