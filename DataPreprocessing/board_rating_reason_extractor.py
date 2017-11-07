
# importing libraries

import numpy as np
import matplotlib as plt
import pandas as pd

# importing the dataset
dataset = pd.read_excel('Training Sheet_wikiurl_budget.xlsx')
# X starts from production_year and omits total
X = dataset.iloc[:, 3:-2].values
# delete the board_rating_reason as it creates too many categories

# Xarr = X.tolist()
y = dataset.iloc[:,12].values

import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords

raw = ' '.join(y)
raw_lower = raw.lower()
tokens = word_tokenize(raw_lower)

# Remove single-character tokens (mostly punctuation)
tokens = [word for word in tokens if len(word) > 1]
text = nltk.Text(tokens)

stopWords = set(stopwords.words('english'))
# print(stopWords)
words_NoStop = []
for w in text:
    if w not in stopWords:
        words_NoStop.append(w)

# print(len(tokens))
# print(tokens)
# colocs = text.collocations(10)
# print(colocs)
top_13_collocations = ['sexual', 'drug', 'thematic', 'sexuality',
                      'disturbing images', 'intense sequences',
                      'humor', 'language',
                      'mature', 'violence','nudity','bloody',
                       'sci-fi','teen','action','General', 'International - to be excluded','smoking']

# text.dispersion_plot(['violence', 'sexual'])
# fdist1 = FreqDist(words_NoStop)
# fdist1.plot(20)
totalcount = 0
for row in y:
    # print(row)
    count = 0
    for colocation in top_13_collocations:

        if row.find(colocation) != -1:
            count += 1
            print('1', end='\t')
        else:
            print('0', end='\t')
    # print(count)
    print('')
    totalcount += count

print(totalcount)
print(len(y))
