#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:28:50 2019

@author: williamlopez
"""

def textArtist(s):
    lyrics=""
    for ind,val in L.iterrows():
        if val["artist"]==s:
            lyrics = lyrics + str(val["lyrics"])
    return lyrics

lyrics = textArtist('arcade-fire')
lyrics = lyrics.replace('\n',' ')
lyrics = lyrics.lower()




doc = nlp(lyrics)
pprint([(X.text, X.label_) for X in doc.ents])

items = [x.text for x in doc.ents]
Counter(items).most_common(3)