#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:56:34 2019

@author: williamlopez
"""

import pandas as pd
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk import tokenize

from tqdm import tqdm_notebook as tqdm
from tqdm import trange

# %%

#functions

def text_emotion(df, column):
    '''
    Takes a DataFrame and a specified column of text and adds 10 columns to the
    DataFrame for each of the 10 emotions in the NRC Emotion Lexicon, with each
    column containing the value of the text in that emotions
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with ten new columns
    '''

    new_df = df.copy()

    filepath = ('project/'
                'NRC-Sentiment-Emotion-Lexicons/'
                'NRC-Emotion-Lexicon-v0.92/'
                'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    emolex_df = pd.read_csv('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
                            names=["word", "emotion", "association"],
                            sep='\t')
    emolex_words = emolex_df.pivot(index='word',
                                   columns='emotion',
                                   values='association').reset_index()
    emotions = emolex_words.columns.drop('word')
    emo_df = pd.DataFrame(0, index=df.index, columns=emotions)

    stemmer = SnowballStemmer("english")

    
    book = ''
    chapter = ''
    
   
    for i, row in new_df.iterrows():
        if (i % 10) == 0:
            print (i, " processed")
            
        if row['Lyrics'] != book:
                #print(row['Lyrics'])
            book = row['Lyrics']
        if row['Year'] != chapter:
                #print('   ', row['Year'])
            chapter = row['Year']
            chap = row['Year']
        document = word_tokenize(new_df.loc[i][column])
        for word in document:
            word = stemmer.stem(word.lower())
            emo_score = emolex_words[emolex_words.word == word]
            if not emo_score.empty:
                for emotion in list(emotions):
                    emo_df.at[i, emotion] += emo_score[emotion]

    new_df = pd.concat([new_df, emo_df], axis=1)

    return new_df


# %%
    
df = pd.read_csv('hip_hop_nocontracted_v4_emotions.csv')
df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

df2 = text_emotion(df, 'Lyrics')


# %%
    
df3 = df2.copy()

emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']

for emotion in emotions:
    df3[emotion] = df3[emotion] / df2['word_count']
    
 
# %%   
    
    
#df2 = df2.drop(['wrd_count2', 'Unnamed: 0.1'], axis=1)

#df2 = df2.rename(columns={'Unnamed: 0':'INDEX'},inplace=True)
    
df3.to_csv("hip_hop_nocontracted_v4_emotions.csv", index_label='INDEX')