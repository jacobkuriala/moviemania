
# importing libraries

import numpy as np
import matplotlib as plt
import pandas as pd

# importing the dataset
dataset = pd.read_excel('Training Sheet.xlsx')
# X starts from production_year and omits total
X = dataset.iloc[:, 3:-2].values
# delete the board_rating_reason as it creates too many categories
X = np.delete(X, 7 , 1)
# Xarr = X.tolist()

y = dataset.iloc[:, 14].values 

# dont need to take care of any missing values

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
# production year
X[:, 0] = labelencoder_x.fit_transform(X[:, 0])
# creative_type
X[:, 2] = labelencoder_x.fit_transform(X[:, 2])
# source
X[:, 3] = labelencoder_x.fit_transform(X[:, 3])
# production_method
X[:, 4] = labelencoder_x.fit_transform(X[:, 4])
# genre
X[:, 5] = labelencoder_x.fit_transform(X[:, 5])
# language
X[:, 6] = labelencoder_x.fit_transform(X[:, 6])
# movie_board_rating_display_name
X[:, 7] = labelencoder_x.fit_transform(X[:, 7])
# movie_release_pattern_display_name
X[:, 8] = labelencoder_x.fit_transform(X[:, 8])

Xarr = X.tolist()

onehotencoder = OneHotEncoder(categorical_features= [0,2,3,4,5,6,7,8])
X = onehotencoder.fit_transform(X).toarray()


