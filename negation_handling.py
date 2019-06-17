#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:17:23 2019

@author: williamlopez
"""

import spacy
import enchant
import pandas as pd
nlp = spacy.load('en_core_web_sm')


from replacers import AntonymReplacer


def negations(text):
    
    sent = text.split()
    noneg = replacer.replace_negations(sent)
    separator = ' '
    out = separator.join(noneg)
    
    return out

replacer = AntonymReplacer()
#replacer.replace('good')
#replacer.replace('uglify')

sent = ['good', 'do', 'not', 'go']
aaa =replacer.replace_negations(sent)

L=pd.read_csv("hip_hop_nocontracted_v4_lowercase.csv", index_col=0)

separator = ' '
bbb = separator.join(aaa)
    
L3=L

L3['Lyrics'] = L3['Lyrics'].apply(negations)


def negations(text):
    
    sent = text.split()
    noneg = replacer.replace_negations(sent)
    separator = ' '
    out = separator.join(noneg)
    
    return out







    