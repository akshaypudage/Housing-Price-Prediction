'''
DATASET : AMES, IOWA HOUSING DATASET
LINK TO DATASET: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/train.csv 

AUTHORS:
DARSHAN KAVATHE & AKSHAY PUDAGE 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn import model_selection as ms
from sklearn.metrics import mean_squared_error, make_scorer
from math import sqrt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def ridgeParameterTuning(X_train,y_train):
    '''
    : This method plots a graph of Results from varying number of alpha vs the RMSE for Ridge Regularized Regression
    :param X_train: train dataset variables
    :param y_train:train dataset target variable
    :return:
    '''
    alphas = [0.1,0.2,1,3,5,10,15,30,60,120]
    ridge = [rmse_cv_train(Ridge(alpha=alpha),X_train,y_train).mean()
             for alpha in alphas]
    ridge = pd.Series(ridge,index=alphas)
    ridge.plot(title="Ridge Regression Parameter Tuning. (Close the plot to continue)")
    plt.xlabel("ALPHA")
    plt.ylabel("RMSE")
    plt.show()


def randomForestParameterTuning(X_train,y_train):
    '''
    : This method plots a graph of Results from varying number of trees vs the RMSE for Random Forest Regression
    :param X_train: train dataset variables
    :param y_train:train dataset target variable
    :return:
    '''
    n_estimators = []
    rmse = []
    for i in range(1,20):
        rf = rmse_cv_train(RandomForestRegressor(n_estimators=i,random_state=40),X_train,y_train).mean()
        n_estimators.append(i)
        rmse.append(rf)
    rf = pd.Series(rmse,index=n_estimators)
    rf.plot(title="RANDOM FOREST PARAMETER TUNING. (Close the plot to continue)")
    plt.xlabel("N_ESTIMATORS")
    plt.ylabel("RMSE")
    plt.show()

def gradientBoostingParameterTuning(X_train,y_train):
    '''
    : This method plots a graph of Results from varying number of trees vs the RMSE for Gradient Boosting Regression.
    :param X_train: train dataset variables
    :param y_train:train dataset target variable
    :return:
    '''
    n_estimators = []
    rmse = []
    for i in range(1,100):
        gb = rmse_cv_train(GradientBoostingRegressor(n_estimators=i,random_state=40),X_train,y_train).mean()
        n_estimators.append(i)
        rmse.append(gb)
    gb = pd.Series(rmse,index=n_estimators)
    gb.plot(title="GRADIENT BOOSTING PARAMETER TUNING. (Close the plot to continue)")
    plt.xlabel("N_ESTIMATORS")
    plt.ylabel("RMSE")
    plt.show()

def ridgePredictionsPlot(X_train,y_train,X_test,y_test):
    '''
    This method plots the graph of actual values of sale price v/s the predicitions obtained from Ridge Regularized Regression.
    :param X_train: train dataset of variables
    :param y_train: train dataset of target variable
    :param X_test: test dataset of variables
    :param y_test: test dataset of target variable
    :return:
    '''
    ridge = Ridge(alpha=15)
    ridge.fit(X_train,y_train)
    train_predictions = ridge.predict(X_train)
    test_predictions = ridge.predict(X_test)
    plt.scatter(train_predictions,y_train,c="blue",marker="s",label = 'Train Data')
    plt.scatter(test_predictions, y_test, c="green", marker="s", label='Test Data')
    plt.title('LINEAR REGRESSION WITH RIDGE REGULARIZATION. (Close the plot to continue)')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.legend(loc="upper left")
    plt.plot([10.5,13.5],[10.5,13.5],c="red")
    plt.show()

def linearRegressionPredictionsPlot(X_train,y_train,X_test,y_test):

    '''
    This method plots the graph of actual values of sale price v/s the predicitions obtained from Linear Regression.
    :param X_train: train dataset of variables
    :param y_train: train dataset of target variable
    :param X_test: test dataset of variables
    :param y_test: test dataset of target variable
    :return:
    '''
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    train_predictions = lr.predict(X_train)
    test_predictions = lr.predict(X_test)
    plt.scatter(train_predictions,y_train,c="blue",marker="s",label = 'Train Data')
    plt.scatter(test_predictions, y_test, c="green", marker="s", label='Test Data')
    plt.title('LINEAR REGRESSION. (Close the plot to continue)')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.legend(loc="upper left")
    plt.plot([10.5,13.5],[10.5,13.5],c="red")
    plt.show()

def findNA(train):
    '''
    This module prints the top 35 missing values in the dataset.
    :param train: the dataset
    :return:
    '''
    print("\nTop 35 variables comprising of missing values")
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missingData = pd.concat([total,percent], axis=1, keys=['Total Missing Values','Percentage of Missing Values'])
    print(missingData.head(35))

def rmse_cv_train(model,X_train,y_train):
    '''
    This function computes the Cross-Validation RMSE for the model on train dataset
    :param model: Machine Learning model.
    :param X_train: Important Variables
    :param y_train: Target Variable
    :return: list of RMSE obtained by performing 10-fold cross-validation.
    '''
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))
    return(rmse)

def main():
    #Path to dataset file
    path_to_dataset = 'train.csv'
    
    # Read the dataset file.
    train = pd.read_csv(path_to_dataset)

    # Compute the correlation heatmap of all variables.
    corrmat = train.corr()
    f, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(corrmat, vmax=0.8, square=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('CORRELATION OF VARIABLES. (Close the plot to continue)')
    sns.plt.show()

    # Check for duplicates in the dataset.
    setID = len(set(train.Id))
    totalIDs = train.shape[0]
    duplicateIDs = totalIDs - setID
    print('Duplicate Training Examples: ' + str(duplicateIDs))

    # Normalize the sale price to reduce skewness
    train.SalePrice = np.log1p(train.SalePrice)
    y = train.SalePrice

    # Print Missing values
    print("---------------------------------------------------------------------------")
    findNA(train)

    # Fill in missing values for the parameters.
    train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
    train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")
    train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("No")
    train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")
    train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
    train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)
    train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
    train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
    train.loc[:, "GarageYrBlt"] = train.loc[:, "GarageYrBlt"].fillna("No")
    train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
    train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
    train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
    train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
    train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
    train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
    train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
    train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
    train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
    train.loc[:, "Electrical"] = train.loc[:, "Electrical"].fillna("No")
    train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("No")
    train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
    train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")
    train.loc[:, "BsmtFinSF2"] = train.loc[:, "BsmtFinSF2"].fillna(0)
    train.loc[:, "BsmtFinSF1"] = train.loc[:, "BsmtFinSF1"].fillna(0)
    train.loc[:, "Exterior1st"] = train.loc[:, "Exterior1st"].fillna("Other")
    train.loc[:, "Exterior2nd"] = train.loc[:, "Exterior2nd"].fillna(train.loc[:, "Exterior1st"])
    train.loc[:, "Exterior2nd"] = train.loc[:, "Exterior2nd"].fillna("Other")
    train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
    train.loc[:, "TotalBsmtSF"] = train.loc[:, "TotalBsmtSF"].fillna(
        train.loc[:, "BsmtFinSF1"] + train.loc[:, "BsmtFinSF2"] + train.loc[:, "BsmtUnfSF"])
    train.loc[:, "SaleType"] = train.loc[:, "SaleType"].fillna("Oth")
    train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")
    train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)
    train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
    train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna(0)
    train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
    train.loc[:, "MSZoning"] = train.loc[:, "MSZoning"].fillna(train["MSZoning"].mode()[0])

    # Replacing numerical features which are actually categorical

    train = train.replace({"MSSubClass": {20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45",
                                          50: "SC50", 60: "SC60", 70: "SC70", 75: "SC75",
                                          80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120",
                                          150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190"},
                           "MoSold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                                      7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
                           })

    # Converting categorical data into ordinal data
    train = train.replace({"Alley": {"Grvl": 1, "Pave": 2},
                           "BsmtCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "BsmtExposure": {"No": 0, "Mn": 1, "Av": 2, "Gd": 3},
                           "BsmtFinType1": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                                            "ALQ": 5, "GLQ": 6},
                           "BsmtFinType2": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                                            "ALQ": 5, "GLQ": 6},
                           "BsmtQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "FireplaceQu": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                                          "Min2": 6, "Min1": 7, "Typ": 8},
                           "GarageCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "GarageQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
                           "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
                           "PavedDrive": {"N": 0, "P": 1, "Y": 2},
                           "PoolQC": {"No": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                           "Street": {"Grvl": 1, "Pave": 2},
                           "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
                           "MasVnrType": {"BrkCmn": 1, "BrkFace": 1, "CBlock": 1,
                                          "Stone": 1, "None": 0},
                           "SaleCondition": {"Abnorml": 0, "Alloca": 0, "AdjLand": 0,
                                             "Family": 0, "Normal": 0, "Partial": 1}}
                          )

    # Converting ordinal data into categorical data.
    train["NewOverallQual"] = train.OverallQual.replace({1: 1, 2: 1, 3: 1,  # bad
                                                           4: 2, 5: 2, 6: 2,  # average
                                                           7: 3, 8: 3, 9: 3, 10: 3  # good
                                                           })
    train["NewOverallCond"] = train.OverallCond.replace({1: 1, 2: 1, 3: 1,  # bad
                                                           4: 2, 5: 2, 6: 2,  # average
                                                           7: 3, 8: 3, 9: 3, 10: 3  # good
                                                           })
    train["NewPoolQC"] = train.PoolQC.replace({1: 1, 2: 1,  # average
                                                 3: 2, 4: 2  # good
                                                 })
    train["NewGarageCond"] = train.GarageCond.replace({1: 1,  # bad
                                                         2: 1, 3: 1,  # average
                                                         4: 2, 5: 2  # good
                                                         })
    train["NewGarageQual"] = train.GarageQual.replace({1: 1,  # bad
                                                         2: 1, 3: 1,  # average
                                                         4: 2, 5: 2  # good
                                                         })
    train["NewFireplaceQu"] = train.FireplaceQu.replace({1: 1,  # bad
                                                           2: 1, 3: 1,  # average
                                                           4: 2, 5: 2  # good
                                                           })
    train["NewFireplaceQu"] = train.FireplaceQu.replace({1: 1,  # bad
                                                           2: 1, 3: 1,  # average
                                                           4: 2, 5: 2  # good
                                                           })
    train["NewFunctional"] = train.Functional.replace({1: 1, 2: 1,  # bad
                                                         3: 2, 4: 2,  # major
                                                         5: 3, 6: 3, 7: 3,  # minor
                                                         8: 4  # typical
                                                         })
    train["NewKitchenQual"] = train.KitchenQual.replace({1: 1,  # bad
                                                           2: 1, 3: 1,  # average
                                                           4: 2, 5: 2  # good
                                                           })
    train["NewHeatingQC"] = train.HeatingQC.replace({1: 1,  # bad
                                                       2: 1, 3: 1,  # average
                                                       4: 2, 5: 2  # good
                                                       })
    train["NewBsmtFinType1"] = train.BsmtFinType1.replace({1: 1,  # unfinished
                                                             2: 1, 3: 1,  # rec room
                                                             4: 2, 5: 2, 6: 2  # living quarters
                                                             })
    train["NewBsmtFinType2"] = train.BsmtFinType2.replace({1: 1,  # unfinished
                                                             2: 1, 3: 1,  # rec room
                                                             4: 2, 5: 2, 6: 2  # living quarters
                                                             })
    train["NewBsmtCond"] = train.BsmtCond.replace({1: 1,  # bad
                                                     2: 1, 3: 1,  # average
                                                     4: 2, 5: 2  # good
                                                     })
    train["NewBsmtQual"] = train.BsmtQual.replace({1: 1,  # bad
                                                     2: 1, 3: 1,  # average
                                                     4: 2, 5: 2  # good
                                                     })
    train["NewExterCond"] = train.ExterCond.replace({1: 1,  # bad
                                                       2: 1, 3: 1,  # average
                                                       4: 2, 5: 2  # good
                                                       })
    train["NewExterQual"] = train.ExterQual.replace({1: 1,  # bad
                                                       2: 1, 3: 1,  # average
                                                       4: 2, 5: 2  # good
                                                       })

    # Creating new features by combining mutually correlated variables.


    train["OverallGrade"] = train["OverallQual"] * train["OverallCond"]
    train["GarageGrade"] = train["GarageQual"] * train["GarageCond"]
    train["ExterGrade"] = train["ExterQual"] * train["ExterCond"]
    train["KitchenScore"] = train["KitchenAbvGr"] * train["KitchenQual"]
    train["FireplaceScore"] = train["Fireplaces"] * train["FireplaceQu"]
    train["GarageScore"] = train["GarageArea"] * train["GarageQual"]
    train["PoolScore"] = train["PoolArea"] * train["PoolQC"]
    train["NewOverallGrade"] = train["NewOverallQual"] * train["NewOverallCond"]
    train["NewExterGrade"] = train["NewExterQual"] * train["NewExterCond"]
    train["NewPoolScore"] = train["PoolArea"] * train["NewPoolQC"]
    train["NewGarageScore"] = train["GarageArea"] * train["NewGarageQual"]
    train["NewFireplaceScore"] = train["Fireplaces"] * train["NewFireplaceQu"]
    train["NewKitchenScore"] = train["KitchenAbvGr"] * train["NewKitchenQual"]
    train["TotalBath"] = train["BsmtFullBath"] + (0.5 * train["BsmtHalfBath"]) + \
                         train["FullBath"] + (0.5 * train["HalfBath"])
    train["AllSF"] = train["GrLivArea"] + train["TotalBsmtSF"]
    train["AllFlrsSF"] = train["1stFlrSF"] + train["2ndFlrSF"]
    train["AllPorchSF"] = train["OpenPorchSF"] + train["EnclosedPorch"] + \
                          train["3SsnPorch"] + train["ScreenPorch"]

    # Computing the correlation of all variables with sale price
    print("---------------------------------------------------------------------------")
    print("\nCorrelation of variables with target variable")
    corr = train.corr()
    corr.sort_values(["SalePrice"], ascending=False, inplace=True)
    print(corr.SalePrice)
    

    print("---------------------------------------------------------------------------")
    print("\nAfter filling missing values.")
    findNA(train)

    # Drop the sale price from the train set as we have already copied it into variable y.
    train = train.drop('SalePrice', 1)

    # Split the dataset into categorical and numerical features
    categorical_features = train.select_dtypes(include=["object"]).columns
    numerical_features = train.select_dtypes(exclude=["object"]).columns
    train_num = train[numerical_features]
    train_cat = train[categorical_features]

    # Logarithmically scale numerical features
    skewness = train_num.apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    skewed_features = skewness.index
    train[skewed_features] = np.log1p(train[skewed_features])

    # Create dummy features for categorical values
    train = pd.get_dummies(train)

    # Split the dataset into training and testing dataset
    X_train, X_test, y_train, y_test = ms.train_test_split(train, y, test_size=0.3, random_state=0)

    # Create machine learning models to predict the sale price.
    models = []
    models.append(('Linear Regression', LinearRegression()))
    models.append(('Ridge Regularized Regression', Ridge(alpha=15)))
    models.append(('Random Forest Regressor', RandomForestRegressor(n_estimators=10, random_state=40)))
    models.append(('Gradient Boosting Regressor',
                   GradientBoostingRegressor(n_estimators=200, random_state=40, learning_rate=0.1)))

    model_names = []
    print("\n===================RESULTS===================\n")

    # For each model, train it using train data, calculate Cross Validation RMSE on Train data and RMSE on test data.
    for name, model in models:
        cv_results = rmse_cv_train(model, X_train, y_train).mean()
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)
        rms = sqrt(mean_squared_error(y_test, y_predictions));
        print(name)
        print("Cross-Validation RMSE (on Train Data) = " +str(cv_results))
        print("RMSE on Test Data = " +str(rms))
        print("---------------------------------------------------")

    #These lines plot the graph of actual v/s predicted value.
    linearRegressionPredictionsPlot(X_train, y_train, X_test, y_test)
    ridgePredictionsPlot(X_train, y_train, X_test, y_test)

    #FOLLOWING LINES PLOT THE PARAMETER-TUNING CURVES FOR EACH OF THE MENTIONED MODELS.

    #UNCOMMENT TO SEE THE PLOTS

    # gradientBoostingParameterTuning(X_train,y_train)
    # randomForestParameterTuning(X_train, y_train)
    # ridgeParameterTuning(X_train,y_train)


if __name__=="__main__":
    main()