#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:14:26 2019

@author: williamlopez
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

import pandas as pd

# %%

#functions

def print_sentiment_scores(sentence):
    snt = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(snt)))
    
    # decide sentiment as positive, negative and neutral 
    if snt['compound'] >= 0.05 :
        return "Positive"
    
    elif snt['compound'] <= - 0.05 : 
        return "Negative" 
  
    else : 
        return "Neutral"
    
    
def print_sentiment_scores_2(sentence):
    snt = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(snt)))
    
    # decide sentiment as positive, negative and neutral 
    return snt['compound']
    





# %%
df = pd.read_csv('hip_hop_nocontracted_v4_lowercase_nonegations.csv')

#df = df[['Lyrics','Year']]
df1 = df.head(1)



df['Sentiment']  = df['Lyrics'].apply(print_sentiment_scores) 

# %%


df['Sentiment_score']  = df['Lyrics'].apply(print_sentiment_scores_2) 


df.to_csv("hip_hop_sentiments.csv")


# %%