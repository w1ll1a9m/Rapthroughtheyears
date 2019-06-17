 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:01:28 2019

@author: williamlopez
"""

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, confusion_matrix

import pandas as pd
import numpy as np 

from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import matplotlib.pyplot as plt

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from pprint import pprint
import re
import itertools
import contractions

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords



nlp = en_core_web_sm.load()




L = pd.read_csv("lyrics.csv", index_col=0)


#dropnan values

L=L.dropna()

#removing end of sentences

L=L.replace({'\n': ' '}, regex=True)

#removing things inside square brackets

L=L.replace(to_replace='\[.*?\]', value='', regex=True)





#further cleaning
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(texto):
    """
        text: a string
        
        return: modified initial string
    """
    texto = texto.lower() # lowercase text
    
    
    texto = contractions.fix(texto)
    texto = re.sub(r"in\' ", "ing ", texto)
    texto = REPLACE_BY_SPACE_RE.sub(' ', texto) # replace REPLACE_BY_SPACE_RE symbols by space in text
    texto = BAD_SYMBOLS_RE.sub('', texto) # delete symbols which are in BAD_SYMBOLS_RE from text
    texto = re.sub(r"verse", "", texto)
    texto = re.sub(r"chorus", "", texto)
    texto = ''.join(''.join(s)[:2] for _, s in itertools.groupby(texto))

    #texto = ' '.join(word for word in texto.split() if word not in STOPWORDS) # delete stopwors from text
    
    return texto


L['lyrics'] = L['lyrics'].apply(clean_text)



#Removing contractions

#L['lyrics'] = L['lyrics'].apply(decontracted)




#getting the word counts
L['word_count'] = L['lyrics'].str.split().str.len()

#removing songs from invalid years
L=L[L['year'] > 1975]

#elimintate the 1-word songs 
L2 = L[L['word_count'] > 50]


L2.to_csv('lyrics_clean.csv', encoding='utf-8')




RapL = L2.loc[L['genre'] == 'Hip-Hop']
RockL = L2.loc[L['genre'] == 'Rock']






# %%




# %%
