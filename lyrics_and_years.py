#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 12:32:45 2019

@author: salihemredevrim
"""

#data sources:  
#https://www.kaggle.com/mousehead/songlyrics/version/1
#https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, confusion_matrix

#%%
#data check

lyrics1 = pd.read_csv('lyrics.csv')
lyrics1['year10'] = lyrics1['year'].apply(lambda x: '70s' if (x >= 1970 and x < 1980) else '80s' if (x >= 1980 and x < 1990) else '90s' if (x >= 1990 and x < 2000) else '00s' if (x >= 2000 and x < 2010) else '10s' if (x >= 2010 and x < 2020) else '0' )

lyrics1['lyrics'] = lyrics1['lyrics'].astype(str)

count1 = lyrics1.groupby('year')['song'].count().reset_index(drop=False)
count2 = lyrics1.groupby(['year','genre']).size().unstack().reset_index(drop=False)

count3 = lyrics1.groupby('artist')['song'].count().reset_index(drop=False)

count4 = lyrics1.groupby(['artist','genre']).size().unstack().reset_index(drop=False)

count4['max']= count4.max(axis=1)

merge1 = pd.merge(count3, count4, how='left', on='artist')

merge1['percent'] = merge1.apply(lambda x: (x['max'] / x['song']) if x.song > 0 else 0, axis=1) 
#all singers assinged to a genre

count5 = merge1[merge1['song'] != merge1['max']]

#%%
del count1, count2, count3, count4, count5, merge1
#%%
#pop, rock, hiphop have been selected 
lyrics2 = lyrics1[lyrics1['year10'] != '0']
lyrics2 = lyrics2[lyrics2['genre'].isin(['Pop', 'Rock', 'Hip-Hop'])]
lyrics3 = lyrics2[['genre', 'year', 'year10', 'lyrics']]

#remove weird chars
lyrics3['lyrics2'] = lyrics3['lyrics'].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

#remove identifiers like chorus, verse, etc
lyrics3['lyrics3'] = lyrics3['lyrics2'].apply(lambda x: re.sub(r'[\(\[].*?[\)\]]', '', str(x)))

lyrics3['len'] = lyrics3.apply(lambda x: len(x['lyrics3'].strip()) if len(x['lyrics3'].strip()) > 0 else 0, axis=1) 
  
lyrics4 = lyrics3[lyrics3['len'] > 50]

lyrics4 = lyrics4[['year', 'year10', 'genre', 'lyrics3']]

#%%
del lyrics1 ,lyrics2, lyrics3

#%%
##check weird characters
#def isEnglish(s):
#    try:
#        s.encode(encoding='utf-8').decode('ascii')
#    except UnicodeDecodeError:
#        return False
#    else:
# 
#        return True
#
##%%    
##lyrics3['check'] = 1
##for k in range(len(lyrics3)):
##    s = lyrics3['lyrics'].iloc[k]
##    lyrics3['check'].iloc[k] = isEnglish(s)

#%%
#check duplicates
check1 = lyrics4['lyrics3'].drop_duplicates()
check2 = lyrics4[['lyrics3', 'year', 'year10', 'genre']].drop_duplicates()

duplicates = lyrics4.lyrics3.value_counts().reset_index(drop=False)
duplicates = duplicates[duplicates['lyrics3'] > 1]

lyrics4['counter'] = lyrics4.groupby('lyrics3').cumcount() + 1

#take the first version
lyrics5 = lyrics4[lyrics4['counter'] == 1]

#%%



