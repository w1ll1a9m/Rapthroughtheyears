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
from replacers import AntonymReplacer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from techniques import *

import string

nlp = en_core_web_sm.load()








L = pd.read_csv("all_songs_data_hip_hop.csv", index_col=0)


#dropnan values

L=L.dropna(subset=['Lyrics'])

#removing end of sentences

L=L.replace({'\n': ''}, regex=True)

#removing things inside square brackets

L=L.replace(to_replace="[\(\[].*?[\)\]]", value='', regex=True)





#further cleaning
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
STOPWORDS = set(stopwords.words('english'))


def negations(text):
    replacer = AntonymReplacer()
    
    sent = text.split()
    noneg = replacer.replace_negations(sent)
    separator = ' '
    out = separator.join(noneg)
    
    return out


def decontracted(phrase):
    # specific
    
    
    
    #phrase = phrase.lower() # lowercase text
    
    
    phrase = re.sub(r",", "", phrase)
    phrase = re.sub(r'i\'mma', 'i am going to', phrase)
    phrase = re.sub(r'i\'ma', 'i am going to', phrase)
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"ain\'t", "are not", phrase)
    phrase = re.sub(r"gonna", "going to", phrase)
    phrase = re.sub(r"wanna", "want to", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r'dammit', 'damn it', phrase)
    phrase = re.sub(r'imma', 'i am going to', phrase)
    phrase = re.sub(r'gimme', 'give me', phrase)
    phrase = re.sub(r'luv', 'love', phrase)
    phrase = re.sub(r' dem ', 'them', phrase)
    phrase = re.sub(r' asap ', 'as soon as possible', phrase)
    phrase = re.sub(r' gyal ', 'girl', phrase)
    phrase = re.sub(r' dat ', ' that ', phrase)
    phrase = re.sub(r' skrrt ', ' ', phrase)
    phrase = re.sub(r' yea ', ' yeah ', phrase)
    phrase = re.sub(r' ayy ', '', phrase)
    phrase = re.sub(r' aye ', '', phrase)
    phrase = re.sub(r' ohoh ', '', phrase)
    phrase = re.sub(r' hol ', 'hold', phrase)
    phrase = re.sub(r' lil ', ' little ', phrase)
    phrase = re.sub(r' g ', ' gangster ', phrase)
    phrase = re.sub(r' gangsta ', ' gangster ', phrase)
    phrase = re.sub(r'thang', 'thing', phrase)
    phrase = re.sub(r'gotta', 'going to', phrase)
    phrase = re.sub(r' hook ', ' ', phrase)
    phrase = re.sub(r' intro ', ' ', phrase)
    phrase = re.sub(r' gon ', ' going to ', phrase)
    phrase = re.sub(r' shoulda ', ' should have ', phrase)
    phrase = re.sub(r' em ', ' them ', phrase)
    phrase = re.sub(r' ya ', ' you ', phrase)
    phrase = re.sub(r' da ', ' the ', phrase)
    phrase = re.sub(r' na na ', ' ', phrase)
    phrase = re.sub(r' hoe', ' whore', phrase)
    phrase = re.sub(r' oh ', ' ', phrase)
    phrase = re.sub(r'\b(\w+)( \1\b)+', r'\1', phrase)
    phrase = re.sub(r'\'til', 'till', phrase)
    phrase = re.sub(r'ooh', '', phrase)
    phrase = re.sub(r'lala', '', phrase)
    phrase = re.sub(r' ho ', ' whore ', phrase)
    phrase = re.sub(r' mm ', '  ', phrase)
    phrase = re.sub(r' yah ', '  ', phrase)
    phrase = re.sub(r' yeah ', '  ', phrase)
    phrase = re.sub(r'hitta', 'nigga', phrase)
    
    
    
    #phrase = re.sub(r'u', 'you', phrase)
    
    
    
    
    phrase = re.sub(r'\&', 'and', phrase)
    phrase = re.sub(r'nothin', 'nothing', phrase)
    phrase = re.sub(r'\$', 's', phrase)
    
    
    phrase = re.sub(r" c\'mon", "come on", phrase)
    phrase = re.sub(r" \'cause", " because", phrase)

    phrase = re.sub(r" cuz ", " because ", phrase)
    phrase = re.sub(r" \'cuz ", " because ", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\'yo", "your", phrase)
    
    
    
    
    return phrase





def clean_text(texto):
    """
        text: a string
        
        return: modified initial string
    """
    texto = texto.lower() # lowercase text
    
    
    
    texto = re.sub(r"in\' ", "ing ", texto)
    texto = REPLACE_BY_SPACE_RE.sub(' ', texto) # replace REPLACE_BY_SPACE_RE symbols by space in text
    texto = BAD_SYMBOLS_RE.sub('', texto) # delete symbols which are in BAD_SYMBOLS_RE from text
    texto = re.sub(r"verse", "", texto)
    #texto = re.sub(r" cause ", " because ", texto)
    texto = re.sub(r"chorus", "", texto)
    texto = ''.join(''.join(s)[:2] for _, s in itertools.groupby(texto))

    texto = ' '.join(word for word in texto.split() if word not in STOPWORDS) # delete stopwors from text
    
    return texto

def removedoublespace(bla):
    
    bla = re.sub(r' +', ' ', bla)
    return bla



L['Lyrics'] = L['Lyrics'].apply(decontracted)   
L['Lyrics'] = L['Lyrics'].apply(clean_text)
L['Lyrics'] = L['Lyrics'].apply(removedoublespace)
L['Lyrics'] = L['Lyrics'].apply(decontracted)
L['Lyrics'] = L['Lyrics'].apply(negations)






#getting the word counts
L['word_count'] = L['Lyrics'].str.split().str.len()

# =============================================================================
# #removing songs from invalid years
# L=L[L['year'] > 1975]
# 
# #elimintate the 1-word songs 
# L2 = L[L['word_count'] > 50]
# 
# 
# L2.to_csv('lyrics_clean.csv', encoding='utf-8')

# RapL = L2.loc[L['genre'] == 'Hip-Hop']
# RockL = L2.loc[L['genre'] == 'Rock']
# =============================================================================











# %%


L2 = L[['Year', 'Artist', 'Song Title', 'Lyrics', 'word_count']]


L2.to_csv("hip_hop_nocontracted_v4_nostopwords.csv")

# %%
