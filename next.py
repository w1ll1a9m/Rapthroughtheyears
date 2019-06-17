#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:00:14 2019

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
import matplotlib as mpl
import seaborn as sns


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from pprint import pprint
import re
import itertools
import contractions

import nltk

from nltk.corpus import stopwords

nlp = en_core_web_sm.load()




# %%

# Functions

def plot_year_counts(dataset, X, title):
    '''Function to plot the year count summaries.
    
    Parameters:

    dataset (dataframe): pandas dataframe
    X (string): the x axis column
    y (string): the y axis column
    title (string): the title of the chart
    
    Outputs a matplotlib lineplot
    '''
    characteristics = dataset.groupby(X).count()
    mpl.rcParams['figure.figsize'] = (35,10,)
    #all_songs.groupby('Year').count().plot(kind='bar')
    sns.barplot(y=characteristics['Artist'], x=characteristics.index)
    plt.title(title)
    plt.ylabel("Number of Songs")
    plt.xticks(rotation=90)



# %%


loaded_song_dataset = pd.read_csv("hip_hop_nocontracted_lowercaseNIGGA.csv", index_col=0)


#RapL = L.loc[L['genre'] == 'Hip-Hop']
#RockL = L.loc[L['genre'] == 'Rock']

songs_with_lyrics_dataset = loaded_song_dataset.dropna(subset=['Lyrics'])

plot_year_counts(loaded_song_dataset, 'Year',  "Number of Songs per Year")
plt.show()



# %%

data1 = loaded_song_dataset[['Lyrics', 'Year']]
data2 = data1[data1['Year'] == 2002]

data2 = data2.applymap(str)


all_text = ' '.join([texto for texto in data2['Lyrics']])

doc = nlp(all_text)

items = [x.text for x in doc.ents]
Counter(items).most_common(10)
