#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 08:35:17 2019

@author: williamlopez
"""

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd

from gensim import corpora, models
stemmer = SnowballStemmer("english")


# %%

#functions


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

# %%
df = pd.read_csv('hip_hop_nocontracted_v4_lowercase_nonegations.csv')

df = df[['Lyrics','Year']]

dfy = df.groupby(by='Year').Lyrics.unique().reset_index()

dfy2 = df.groupby(by='Year').agg(lambda col: ''.join(col)).reset_index()



processed_docs = df['Lyrics'].map(preprocess)


dictionary = gensim.corpora.Dictionary(processed_docs)


# %%

dictionary.filter_extremes(keep_n=10000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

bow_doc_4310 = bow_corpus[10]

for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                               dictionary[bow_doc_4310[i][0]], 
bow_doc_4310[i][1]))
    
    
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break
    

# %%

#models LDA LSI gensim

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=2, workers=2)

print('\n LDA BOW \n') 

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    
print('\n LDA TFIDF \n') 
 
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
    
print('\n LSI BOW \n')
lsi_model = models.LsiModel(bow_corpus, num_topics=5, id2word=dictionary)

for idx, topic in lda_model_tfidf.print_topics(-1):
   print('Topic: {} Word: {}'.format(idx, topic))
  
print('\n LSI TFIDF \n')
lsi_model_tfidf = models.LsiModel(corpus_tfidf, num_topics=5, id2word=dictionary)

for idx, topic in lda_model_tfidf.print_topics(-1):
   print('Topic: {} Word: {}'.format(idx, topic))
   
   
   # %%
   
#models lds lsi nmf scikitlearn


NUM_TOPICS = 3
 
vectorizer = CountVectorizer(min_df=5, max_df=0.9, 
                             stop_words='english', lowercase=True, 
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(df['Lyrics'])
 
# Build a Latent Dirichlet Allocation Model
lda_model_sci = LatentDirichletAllocation(n_topics=NUM_TOPICS, max_iter=10, learning_method='online')
lda_Z = lda_model_sci.fit_transform(data_vectorized)
print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
 
# Build a Non-Negative Matrix Factorization Model
nmf_model_sci = NMF(n_components=NUM_TOPICS)
nmf_Z = nmf_model_sci.fit_transform(data_vectorized)
print(nmf_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
 
# Build a Latent Semantic Indexing Model
lsi_model_sci = TruncatedSVD(n_components=NUM_TOPICS)
lsi_Z = lsi_model_sci.fit_transform(data_vectorized)
print(lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
 
 
# Let's see how the first document in the corpus looks like in different topic spaces
print(lda_Z[0])
print(nmf_Z[0])
print(lsi_Z[0])
   


 
print("LDA Model:")
print_topics(lda_model_sci, vectorizer)
print("=" * 20)
 
 
print("LSI Model:")
print_topics(lsi_model_sci, vectorizer)
print("=" * 20)

print("NMF Model:")
print_topics(nmf_model_sci, vectorizer)
print("=" * 20) 
   
   
   
   
   
   
   
   
   
   
# %%  

#scikit learn v2

no_features = 1000


no_topics = 10

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(df['Lyrics'])
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(df['Lyrics'])
tf_feature_names = tf_vectorizer.get_feature_names()



#LSI tfidf

lsi_model_sci_tfidf = TruncatedSVD(n_components=no_topics)
lsi_tfidf = lsi_model_sci_tfidf.fit_transform(tfidf)
lsi_tfidf_feature_names = tf_vectorizer.get_feature_names()


#LSI tfidf

lsi_model_sci_count = TruncatedSVD(n_components=no_topics)
lsi_count = lsi_model_sci_count.fit_transform(tf)
lsi_count_feature_names = tf_vectorizer.get_feature_names()


# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 15

print ("TFIDF NMF \n")
display_topics(nmf, tfidf_feature_names, no_top_words)
print("\n")
print ("LDA \n")
display_topics(lda, tf_feature_names, no_top_words)
print("\n")
print ("TFIDF LSI \n")
display_topics(lsi_tfidf, lsi_tfidf_feature_names, no_top_words)
print("\n")
print ("COUNT LSI \n")
display_topics(lsi_count, lsi_count_feature_names, no_top_words)
print("\n")



