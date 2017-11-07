
# importing libraries

import numpy as np
import matplotlib as plt
import pandas as pd

# importing the dataset
dataset = pd.read_excel('Training Sheet_budget.xlsx')
# X starts from production_year and omits total
X = dataset.iloc[:, 3:-2].values
# delete the board_rating_reason as it creates too many categories
X = np.delete(X, 8 , 1)
# deleting language because it seems to reduce the variability of the data 
X = np.delete(X, 7 , 1)
X = np.delete(X, 1 , 1)
# Xarr = X.tolist()

y = dataset.iloc[:, 15].values 

#take care of any missing values in production budget
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:1])
X[:, 0:1] = imputer.transform(X[:, 0:1])

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
# production year
X[:, 2] = labelencoder_x.fit_transform(X[:, 2])
# creative_type
X[:, 3] = labelencoder_x.fit_transform(X[:, 3])
# source
X[:, 4] = labelencoder_x.fit_transform(X[:, 4])
# production_method
X[:, 5] = labelencoder_x.fit_transform(X[:, 5])
# genre
X[:, 6] = labelencoder_x.fit_transform(X[:, 6])
# language
X[:, 7] = labelencoder_x.fit_transform(X[:, 7])
# movie_board_rating_display_name
# X[:, 8] = labelencoder_x.fit_transform(X[:, 8])
# movie_release_pattern_display_name
# X[:, 9] = labelencoder_x.fit_transform(X[:, 9])

# Xarr = X.tolist()

onehotencoder = OneHotEncoder(categorical_features= [2,3,4,5,6,7])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# feature Scaling - TODO: may have to feature scale the output
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# y_train = sc.fit_transform(y_train)
# y_test = sc.transform(y_test)
# X = sc.fit_transform(X)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import matplotlib.pylab as plt


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

# train = pd.read_csv('train_modified.csv')
# target = 'Disbursed'
# IDcol = 'ID'

def modelfit(alg, ip, op,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        print(ip.shape)
        print(op.shape)
        op = op.astype(int)
        xgtrain = xgb.DMatrix(ip, label=op.astype(int))
        print(xgtrain.get_label())
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            early_stopping_rounds=early_stopping_rounds)
        
        print(cvresult)
        # metrics='auc', 
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(ip, op,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(ip)
    dtrain_predprob = alg.predict_proba(ip)[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(op, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(op, dtrain_predprob))
      
    # alg.Booster().get_fscore()       
    # feat_imp = pd.Series().sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')


#Choose all predictors except target & IDcols
'''
# trying to find optimal tree count
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class=9,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, X, y)
'''
param_test1 = {
 'max_depth':range(3,10,3),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', num_class=4, nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5, verbose = 2)
gsearch1.fit(X,y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_