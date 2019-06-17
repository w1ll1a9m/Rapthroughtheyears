#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:09:16 2019

@author: williamlopez
"""

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
#import mpld3

from sklearn.feature_extraction.text import TfidfVectorizer


# %%


df = pd.read_csv('hip_hop_topic_modeling10.csv')

df = df[['Year','processed_lyrics']]

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features=1000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_lyrics']) #fit the vectorizer to lyrics

print(tfidf_matrix.shape)
