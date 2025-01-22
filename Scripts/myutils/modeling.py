from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def linear_regression(X_train,y_train):
    linear_mod = LinearRegression()
    linear_mod.fit(X_train,y_train)
    return linear_mod

def ridge_regression(X_train,y_train,alpha = 1.0):
    ridgemodel = Ridge(alpha = alpha)
    ridgemodel.fit(X_train,y_train)
    return ridgemodel

def lasso_regression(X_train,y_train,alpha = 1.0):
    lasso_mod = Lasso(alpha = alpha)
    lasso_mod.fit(X_train,y_train)
    return lasso_mod

def sgd_regression (X_train,y_train, random_state):
    sgd_reg = SGDRegressor(
        loss = 'squared_loss',
        max_iter = 1000,
        tol = 1e-3,
        random_state = random_state)
    
    sgd_reg.fit(X_train,y_train)
    return sgd_reg

def dt_regression(X_train,y_train,random_state):
    dt_mod = DecisionTreeRegressor(
        max_depth = None,
        random_state = random_state
    )
    dt_mod.fit(X_train,y_train)
    return dt_mod
    
def ablation_experiments(X_train, X_test, y_train, y_test,mod):
    model = mod

    mse_results = {}

    for col in X_train.columns:
        X_train_ablated = X_train.drop(columns=[col])
        X_test_ablated = X_test.drop(columns=[col])

        model.fit(X_train_ablated, y_train)

        y_pred_ablated = model.predict(X_test_ablated)

        mse_ablated = mean_squared_error(y_test, y_pred_ablated)
        mse_results[col] = np.round(mse_ablated, 4)

    sorted_mse_results = dict(sorted(mse_results.items(), key=lambda item: item[1]))

    for col, mse_ablated in sorted_mse_results.items():
        print(f"Removed Attribute: {col}, Test MSE: {mse_ablated}")
    return sorted_mse_results

def cross_val (mod,X,Y,num_folds):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(mod, X, Y, cv=kf, scoring='accuracy')
    
    print("Mean Accuracy:", np.round(cv_scores.mean()*100,2),'%')
    
def decisiontree (X_train,y_train,rs):
    dt = DecisionTreeClassifier(criterion = "gini",random_state = rs,min_samples_leaf = 3)
    dt.fit(X_train,y_train)
    return dt

def randomforest (X_train,Y_train,rs):
    rf = RandomForestClassifier(criterion = "gini",random_state = rs,min_samples_leaf = 3)
    rf.fit(X_train,Y_train)
    return rf
    
def KNearstNeighbor (X_train,Y_train):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, Y_train)
    return knn_model 

def classificationReport(mod, X_test,y_test):
    Y_fit = mod.predict(X_test)
    print(classification_report(y_test,Y_fit))

    
    

