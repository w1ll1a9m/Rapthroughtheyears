#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:42:27 2019

@author: williamlopez
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist

nltk.download('punkt')

from nltk import word_tokenize,sent_tokenize


# %%
df = pd.read_csv('hip_hop_nocontracted_lowercase_v3_no_stopwords.csv')

df = df[['Lyrics','Year']]

df['word_count'] = df['Lyrics'].str.split().str.len()


#sns.violinplot(x=df["word_count"])


# %%

#number of words per year

plt.rc("figure", figsize=(12, 6))
ax =sns.boxplot(x="Year", y="word_count", data=df)

ax.set(ylim=(10, 1000))


# %%

# most frequent words

customStopWords = ["'s", "n't", "'m", "'re", "'ll","'ve","...", "ä±", "''", '``',\
                  '--', "'d", 'el', 'la']
stopWords = stopwords.words('english') + customStopWords

words = ""
for song in df.iterrows():
    words += " " + song[1]['Lyrics']

words = nltk.word_tokenize( words.lower() )
words = [ word for word in words if len(word) > 1\
                             and not word.isnumeric()\
                             and word not in stopWords ]
    
word_dist = FreqDist( words  )
print("The 10 most common words in the dataset are :")
for word, frequency in word_dist.most_common(20):
    print( u'{} : {}'.format( word, frequency ) )

plt.figure(figsize=(15, 10))
nlp_words = word_dist.plot( 20 )