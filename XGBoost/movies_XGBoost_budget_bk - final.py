
# importing libraries

import numpy as np
import matplotlib as plt
import pandas as pd

# importing the dataset
dataset = pd.read_excel('Training Sheet_budget_bk.xlsx')
# X starts from production_year and omits total
X = dataset.iloc[:, 3:-2].values
# delete the board_rating_reason as it creates too many categories
X = np.delete(X, 8 , 1)
# deleting language because it seems to reduce the variability of the data 
X = np.delete(X, 7 , 1)
# delete year
X = np.delete(X, 1 , 1)
# delete movie_seq
X = np.delete(X, 1 , 1)

# importing the dataset
dataset2 = pd.read_excel('Scoring Sheet_budget.xlsx')
# X starts from production_year and omits total
X2 = dataset2.iloc[:, 3:].values
# delete the board_rating_reason as it creates too many categories
X2 = np.delete(X2, 8 , 1)
# deleting language because it seems to reduce the variability of the data 
X2 = np.delete(X2, 7 , 1)
# delete year
X2 = np.delete(X2, 1 , 1)
# delete movie_seq
X2 = np.delete(X2, 1 , 1)

# Xarr = X.tolist()

y = dataset.iloc[:, 15].values 

#take care of any missing values in production budget
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:1])
X[:, 0:1] = imputer.transform(X[:, 0:1])
X2[:,0:1] = imputer.transform(X2[:, 0:1])

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:, 1] = labelencoder_x.fit_transform(X[:, 1])
X2[:, 1] = labelencoder_x.transform(X2[:, 1])
# production year
X[:, 2] = labelencoder_x.fit_transform(X[:, 2])
X2[:, 2] = labelencoder_x.transform(X2[:, 2])
# creative_type
X[:, 3] = labelencoder_x.fit_transform(X[:, 3])
X2[:, 3] = labelencoder_x.transform(X2[:, 3])
# source
X[:, 4] = labelencoder_x.fit_transform(X[:, 4])
X2[:, 4] = labelencoder_x.transform(X2[:, 4])
# production_method
X[:, 5] = labelencoder_x.fit_transform(X[:, 5])
X2[:, 5] = labelencoder_x.transform(X2[:, 5])
# genre
X[:, 6] = labelencoder_x.fit_transform(X[:, 6])
X2[:, 6] = labelencoder_x.transform(X2[:, 6])

onehotencoder = OneHotEncoder(categorical_features= [1,2,3,4,5,6])
X = onehotencoder.fit_transform(X).toarray()

X2 = onehotencoder.transform(X2).toarray()

# feature Scaling - TODO: may have to feature scale the output
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(X)
X2 = sc.transform(X2)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(objective='multi:softmax')
classifier.fit(X,y)

# Predicting the Test set results
y_pred = classifier.predict(X2)

