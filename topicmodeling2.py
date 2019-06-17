#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:01:29 2019

@author: williamlopez
"""


from __future__ import print_function
# Usual imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
import concurrent.futures
import time
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans
import os
import re
from nltk.tokenize import RegexpTokenizer

import nltk
import matplotlib.pyplot as plt
import matplotlib as mpl
from nltk.stem.snowball import SnowballStemmer
from sklearn.manifold import MDS
#print(os.listdir("../"))

# Plotly based imports for visualization
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

nlp = spacy.load('en_core_web_sm')

from sklearn.metrics.pairwise import euclidean_distances

from gensim import corpora, models

from sklearn.manifold import TSNE

from pylab import rcParams

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
#%matplotlib inline
import seaborn as sns
# Bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
from bokeh.layouts import column
from bokeh.palettes import all_palettes
output_notebook()


import seaborn as sns

# %%
#functions


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens




# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]]) 
    

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

# %%

#df = pd.read_csv('hip_hop_nocontracted_v4_lowercase_nonegations.csv')

df = pd.read_csv('hip_hop_nocontracted_v4_nostopwords.csv')


#df = df[['Lyrics','Year']]

doc = nlp(df['Lyrics'][3])

stopwords = list(STOP_WORDS)
punctuations = string.punctuation


parser = English()

tqdm.pandas()
df["processed_lyrics"] = df["Lyrics"].progress_apply(spacy_tokenizer)

# %%
no_features = 1000


no_topics = 3

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(df['processed_lyrics'])
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=0.2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(df['processed_lyrics'])
tf_feature_names = tf_vectorizer.get_feature_names()



#LSI tfidf

lsi_model_sci_tfidf = TruncatedSVD(n_components=no_topics)
lsi_tfidf = lsi_model_sci_tfidf.fit_transform(tfidf)


#LSI tfidf

lsi_model_sci_count = TruncatedSVD(n_components=no_topics)
lsi_count = lsi_model_sci_count.fit_transform(tf)


# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
nmf_tf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tf)
# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)






no_top_words = 10

print ("TFIDF NMF \n")
display_topics(nmf, tfidf_feature_names, no_top_words)
print("\n")
print ("BOW NMF \n")
display_topics(nmf_tf, tf_feature_names, no_top_words)
print("\n")

print ("BOW LDA \n")
display_topics(lda, tf_feature_names, no_top_words)
print("\n")




 #=============================================================================
 #
print("LSI TFIDF:")
print_topics(lsi_model_sci_tfidf, tfidf_vectorizer)
print("=" * 20)
print("LSI tf:")
print_topics(lsi_model_sci_count, tf_vectorizer)
print("=" * 20)
 #=============================================================================

# %%

tfidf2 = pd.DataFrame(tfidf.toarray())

nmf_Z = nmf.fit_transform(tfidf)
#, 'Topic4':nmf_Z[:,3], 'Topic 5':nmf_Z[:,4]
datasetx = pd.DataFrame({'Topic1':nmf_Z[:,0],'Topic2':nmf_Z[:,1], 'Topic3':nmf_Z[:,2]})

#dash = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer, mds='tsne')


pdx = pd.concat([df, datasetx], axis=1, sort=False)

pdx = pdx.drop(['Unnamed: 0'], axis=1)

pdx.to_csv("hip_hop_topic_modeling10.csv",index_label='INDEX')



# %%



#getting the topic words

stopwords = nltk.corpus.stopwords.words('english')


stemmer = SnowballStemmer("english")


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in df['Lyrics']:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')



# %%

#define vectorizer parameters
tfidf_stem = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_stem.fit_transform(df['Lyrics']) #fit the vectorizer to synopses

print(tfidf_matrix.shape)




# %%
#some clustering

#distance between lyrics
dist = 1 - cosine_similarity(tfidf)


num_clusters = 3

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf)

clusters = km.labels_.tolist()



# %%


pdclusters = pd.DataFrame(clusters)

pdclusters = pdclusters.rename(columns={0 : "Cluster"})

df2 = pd.DataFrame()

df2['Year'] = df['Year']
df2['Song Title'] = df['Song Title']

dfcl = pd.concat([df2,pdclusters], axis=1, sort=False)
dfcl.to_csv("hip_hop_kmeansclustering.csv",index_label='INDEX')

#dfcl['Cluster'].value_counts()



# %%

#clustering

terms = tfidf_stem.get_feature_names()


print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    
    #print("Cluster %d titles:" % i, end='')
    #for title in dfcl.ix[i]['Artist'].values.tolist():
        #print(' %s,' % title, end='')
    #print() #add whitespace
    #print() #add whitespace
    
print()
print()



# %%



MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]



# %%


#set up colors per clusters using a dict
cluster_colors = {0: '#ff5667', 1: '#060360', 2: '#b7e1fb', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'Cluster 1: life, big, nights ', 
                 1: 'Cluster 2: feel, hit, life ', 
                 2: 'Cluster 3: make, baby, turn ', 
                 3: 'Cluster 4: baby, stay, home ', 
                 4: 'Cluster 5: better, man, mind '}
                 

#create data frame that has the result of the MDS plus the cluster numbers and titles
dfa = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title = '')) 

#group by cluster
groups = dfa.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(dfa)):
    ax.text(dfa.ix[i]['x'], dfa.ix[i]['y'], dfa.ix[i]['title'], size=8)  

    
    
plt.show() #show the plot

# to save the plottt
#plt.savefig('clusters_small_noaxes.png', dpi=200)


# %%

#heatmap
rcParams['figure.figsize'] = 10, 10
def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()


#test_array = np.arange(100 * 100).reshape(100, 100)
heatmap2d(dist)