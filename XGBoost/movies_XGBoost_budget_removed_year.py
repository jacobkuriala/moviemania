
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
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# feature Scaling - TODO: may have to feature scale the output
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# y_train = sc.fit_transform(y_train)
# y_test = sc.transform(y_test)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(objective='multi:softmax')
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies)
print(accuracies.mean())
print(accuracies.std())