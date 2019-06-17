#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:04:11 2019

@author: salihemredevrim
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import spacy
from wordcloud import WordCloud
from collections import Counter
import en_core_web_sm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import gensim.models.word2vec as word2vec
import multiprocessing
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.test.utils import get_tmpfile
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn import utils
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
#%%
#Some analyses

#datasets
hip_hop = pd.read_csv('all_songs_data_hip_hop_clean.csv')
rock = pd.read_csv('all_songs_data_rock_clean.csv')
keep_list = ['Year', 'Artist', 'Song Title', 'Lyrics', 'word_count']

#%%
def some_analyses(data1, word_cutoff):
#first, a bit more preprocessing 
#there are some duplicates since some songs were listed as top in consecutive years 
#check duplicates
#check1 = data1['Lyrics'].drop_duplicates()
#check2 = data1[keep_list].drop_duplicates()

    duplicates = data1.Lyrics.value_counts().reset_index(drop=False)
    duplicates = duplicates[duplicates['Lyrics'] > 1]

    data1['counter'] = data1.groupby('Lyrics').cumcount() + 1

    #take the first version
    data1 = data1[data1['counter'] == 1]

    #outlier elimination 
    data1 = data1[data1['word_count'] <= word_cutoff]
    
    data1 = data1[keep_list]

    count1 = data1.groupby('Year')['Lyrics'].count().reset_index(drop=False)
    count2 = data1.groupby('Artist')['Lyrics'].count().reset_index(drop=False).sort_values('Lyrics', ascending=False).head(20)
    count3 = data1.groupby('Year')['word_count'].mean().reset_index(drop=False)

    sns.barplot(y=count1.Lyrics, x=count1.Year)
    plt.title('number of songs per year')
    plt.ylabel('Number of Songs')
    plt.xticks(rotation=45)
    plt.show()
    
    sns.barplot(y=count3.word_count, x=count3.Year)
    plt.title('average word count per year')
    plt.ylabel('Word Counts')
    plt.xticks(rotation=45)
    plt.show()

    return data1, count2

#%%
    
hip_hop, singer_counts = some_analyses(hip_hop, 2000)
rock, singer_counts = some_analyses(rock, 2000)   

#sample = hip_hop.head(200)
#%%
#Maybe more data with genres!? Around 1100 more songs keep in mind
#Not added for now 
#actually only important thing is singer for old data since each singer was assigned to a specific genre
billboard_new = pd.read_excel('Billboard_64_15.xlsx')
old_data = pd.read_csv('lyrics.csv') 

#old_data = old_data[['song', 'artist', 'genre', 'year']]
old_data1 = old_data[['artist', 'genre']].drop_duplicates()

check1 = old_data1['artist'].drop_duplicates()
#same
drop_list = ['Not Available', 'Other']
old_data1  = old_data1[~old_data1['genre'].isin(drop_list)]

#old_data['song'] = old_data['song'].str.replace('-',' ')
old_data1['artist'] = old_data1['artist'].str.replace('-',' ')

billboard_new1 = pd.merge(billboard_new, old_data1, how='left', left_on=['Artist'], right_on=['artist'])

billboard_new1 = billboard_new1[billboard_new1['genre'].notnull()]

#another sample for NER
sample2 = old_data.head(300)

#%%

def spacy_data(data1, lyrics):

    #find verb, adverb, noun, and stop words
    #init 
    verbs = []
    nouns = []
    adverbs = []
    corpus = []
    nlp = spacy.load('en_core_web_sm')
    #nlp = en_core_web_sm.load()
    
    for i in range (0, len(data1)):
        #print("song: ", i, end = "\n")
        song = data1.iloc[i][lyrics]
        doc = nlp(song)
        spacy_data = pd.DataFrame()
        
        for token in doc:
            if token.lemma_ == "-PRON-":
                    lemma = token.text
            else:
                lemma = token.lemma_
            row = {
                "Word": token.text,
                "Lemma": lemma,
                "PoS": token.pos_,
                "Stop Word": token.is_stop
            }  
            spacy_data = spacy_data.append(row, ignore_index = True)
            
        verbs.append(" ".join(spacy_data["Lemma"][spacy_data["PoS"] == "VERB"].values))
        nouns.append(" ".join(spacy_data["Lemma"][spacy_data["PoS"] == "NOUN"].values))
        adverbs.append(" ".join(spacy_data["Lemma"][spacy_data["PoS"] == "ADV"].values))
        corpus_clean = " ".join(spacy_data["Lemma"][spacy_data["Stop Word"] == False].values)
        corpus_clean = re.sub(r'[^A-Za-z0-9]+', ' ', corpus_clean)   
        corpus.append(corpus_clean)
        
    data1['Verbs'] = verbs
    data1['Nouns'] = nouns
    data1['Adverbs'] = adverbs
    data1['Corpus'] = corpus
    
    return data1

#%%
dataset = spacy_data(hip_hop, 'Lyrics'); 

#%%
#What about bigrams?

#%%
def word_counts(data1, pos, year, most_num):
   
    # init
    freq = pd.DataFrame()
    common_words = []
    years = data1[year].unique().tolist()
    
    #frequencies per each year
    for i in range (0, len(years)):
        year_corpus = str(dataset[pos][data1[year] == years[i]].tolist())
        tokens = year_corpus.split(" ")
        counts = Counter(tokens)
        freq = freq.append({
            "Year": years[i],
            "Terms": counts.most_common(n=most_num)
        }, ignore_index=True)
    freq['Year'] = freq['Year'].astype(int)
    
    #distinct words through years 
    for i in range (0, len(freq)): 
        for words in freq['Terms'][i]:
            common_words.append(words[0])
            
    common_words = list(set(common_words))
    
    #tabularize
    data2 = pd.DataFrame(dict.fromkeys(common_words, [0]))
    data2['Year'] = 0
    data3 = data2.copy()
    
    for j in freq['Year']:
        row1 = data2.copy()
        row1['Year'] = j 
        data3 = data3.append(row1)
    
    data3 = data3[1:]
    data3 = data3.reset_index(drop=True)    
    
    
    for j in range(0, len(data3)):
            current_year = freq['Year'][j]
            current_terms = freq['Terms'][j]
            
            for words in current_terms:
                data3[words[0]] = data3.apply(lambda x: words[1] if x['Year'] == current_year else x[words[0]], axis=1)
  
    return freq, common_words, data3

#%%
    
frequencies, common_words, dataset2 = word_counts(dataset, 'Nouns', 'Year', 50)   

#%%    
def ner(data1, lyrics):

#CASE SENSITIVE (?)    
#https://medium.com/@dudsdu/named-entity-recognition-for-unstructured-documents-c325d47c7e3a   
    nlp = spacy.load('en_core_web_sm')
    #nlp = en_core_web_sm.load()
    data1['PERSON'] = ''; 
    data1['NORP'] = ''; 
    data1['FAC'] = ''; 
    data1['ORG'] = ''; 
    data1['GPE'] = ''; 
    data1['LOC'] = ''; 
    data1['PRODUCT'] = ''; 
    data1['EVENT']  = ''; 
    data1['WORK_OF_ART'] = ''; 
    
    for i in range(0, len(data1)):

#WE CAN MERGE SOME, ALSO A BIT SLOW COULDNT HANDLE WEIRD DT     
        
        
#PERSON	People, including fictional.
#NORP	Nationalities or religious or political groups.
#FAC	Buildings, airports, highways, bridges, etc.
#ORG	Companies, agencies, institutions, etc.
#GPE	Countries, cities, states.
#LOC	Non-GPE locations, mountain ranges, bodies of water.
#PRODUCT	Objects, vehicles, foods, etc. (Not services.)
#EVENT	Named hurricanes, battles, wars, sports events, etc.
#WORK_OF_ART	Titles of books, songs, etc.
        
        person = '';
        norp ='';
        fac = '';
        org = '';
        gpe = '';
        loc = '';
        product = ''; 
        event = '';
        work_of_art = '';
        
        song = data1.iloc[i][lyrics]
        doc = nlp(song)
        ents = [(e.text, e.label_) for e in doc.ents]
        ents1 = pd.DataFrame(ents)
        
        for k in range(0, len(ents1)):
            
            if ents1.iloc[k][1] == 'PERSON': 
              person = person+ ', '+ents1.iloc[k][0]
            elif ents1.iloc[k][1] == 'NORP':
              norp =  norp+', '+ents1.iloc[k][0]
            elif ents1.iloc[k][1] == 'FAC':
              fac =  fac+', '+ents1.iloc[k][0]
            elif ents1.iloc[k][1] == 'ORG':
              org =  org+', '+ents1.iloc[k][0]
            elif ents1.iloc[k][1] == 'GPE':
              gpe =  gpe+', '+ents1.iloc[k][0]
            elif ents1.iloc[k][1] == 'LOC':
              loc =  loc+', '+ents1.iloc[k][0]
            elif ents1.iloc[k][1] == 'PRODUCT':
              product =  product+', '+ents1.iloc[k][0]
            elif ents1.iloc[k][1] == 'EVENT':
              event =  event+', '+ents1.iloc[k][0]
            elif ents1.iloc[k][1] == 'WORK_OF_ART':
              work_of_art = work_of_art+', '+ents1.iloc[k][0] 
              
               
        data1['PERSON'].iloc[i] = person; 
        data1['NORP'].iloc[i] = norp; 
        data1['FAC'].iloc[i] = fac;
        data1['ORG'].iloc[i] = org;
        data1['GPE'].iloc[i] = gpe;
        data1['LOC'].iloc[i] = loc;
        data1['PRODUCT'].iloc[i] = product;
        data1['EVENT'].iloc[i] = event;
        data1['WORK_OF_ART'].iloc[i] = work_of_art;
    
    return data1

#%%

data123 = ner(sample2, 'lyrics')

#%%
#How can write labels inside like examples in slides?!

#data3 = data2.melt('Year', var_name='cols',  value_name='Words').sort_values('Year').reset_index(drop=True)
#sns.factorplot(x='Year', y="Words", hue='cols', data=data3, stacked=True)
#plt.show()

def plotting(dataset2, num_words):
    
    #make year index and take most popular n words
    data3 = dataset2.set_index('Year') 
    sum_over = data3.sum(axis = 0, skipna = True).reset_index(drop=False).sort_values(0, ascending=False).reset_index(drop=True)
    data4 = sum_over.head(num_words) 
    keep_list = data4['index'].tolist()
    data5 = data3[keep_list] 

    #plot
    plt.style.use('seaborn')
    data5.plot.area()
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Frequent Words', fontsize=15)
    plt.title('Most Frequent Words Through Years',fontsize=15)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0], reverse=True))
    ax.legend(handles, labels)
    plt.show()
    
    return;

#%%
plotting(dataset2, 10)    
    
#%%

def wordcloud(freq, year, pos, max_words):
    
    freq1 = freq[freq['Year'] == year].reset_index(drop=True)
    freq2 = pd.DataFrame(freq1['Terms'][0]).astype(str)
    freq2 = freq2.rename(index=str, columns={0: 'word', 1: 'count'})
    
    d = {}
    for a, x in freq2.values:
        d[a] = float(x)

    wordcloud = WordCloud( width = 4000,
                          height = 3000,
                          background_color="white",
                          max_words = max_words )
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure()
    
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    return;
    
#%%

wordcloud(frequencies, 2018, 'Nouns', 50)
wordcloud(frequencies, 2017, 'Nouns', 50)
    
#%%
#Heatmap 

def heatmappp(dataset2, num_words): 
    
    #make year index and take most popular n words
    heatmap = dataset2.set_index('Year')
    sum_over = heatmap.sum(axis = 0, skipna = True).reset_index(drop=False).sort_values(0, ascending=False).reset_index(drop=True)
    heatmap2 = sum_over.head(num_words) 
    keep_list = heatmap2['index'].tolist()
    heatmap3 = heatmap[keep_list] 

    plt.figure(figsize=(15,15))
    sns.heatmap(data=heatmap3)
    plt.show()    
    return; 
    
#%%
heatmappp(dataset2, 15)   

#%%
#Do we care Profanity analysis?? 

#Also topic extraction seems weird 
#https://michaeljohns.github.io/lyrics-lab/#grp-nlp-vocab

#%%
#Simple rock vs hiphop tfidf etc 
#hiphop: 1 rock: 0 
#also we can try with nouns, verbs etc 
#Do we want to do grid search, I think overfitting to training set is fine also we have parameters for tfidf 

def tfidf_so_on(data1, data2, text, min_df, max_df, ngram_range1, ngram_range2): 
    #Data1: target 1 data 
    #Data2: target 0 data 
    #text: column for text    
    
    #Data preparation     
    data1 = pd.DataFrame(data1[text])
    data1['Target'] = 1

    data2 = pd.DataFrame(data2[text])
    data2['Target'] = 0
    
    #Balance 
    min1 = min(len(data1), len(data2));
    
    
    data1 = data1.sample(n=min1, random_state=1905)
    data2 = data2.sample(n=min1, random_state=1905)
    
    data_all = data1.append(data2, ignore_index=True)

    #train - test split 
    X_train, X_test, y_train, y_test = train_test_split(data_all[text], data_all['Target'], stratify=data_all['Target'], test_size=0.2, random_state=1905)

    #tokenization - bow
    vect = CountVectorizer().fit(X_train)
    #check1 = vect.get_feature_names()

    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    #initial models
    #Logistic Regression
    model1 = LogisticRegression()
    model1.fit(X_train_vectorized, y_train)

    #Predict on test set
    predictions = model1.predict(X_test_vectorized)

    #score
    roc1 = roc_auc_score(y_test, predictions)
    accuracy1 = accuracy_score(y_test, predictions)
    precision1 = precision_score(y_test, predictions)
    recall1 = recall_score(y_test, predictions)
    f1_score1 = f1_score(y_test, predictions)
    
    
    #F1 = 2 * (precision * recall) / (precision + recall)
    
    #print('AUC: ', roc_auc_score(y_test, predictions))
    #false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
    #roc_auc = auc(false_positive_rate, true_positive_rate)
    #print('accuracy: ', accuracy_score(y_test, predictions))
    #confusion = confusion_matrix(y_test, predictions)

    #svm
    model2 = SVC()
    model2.fit(X_train_vectorized, y_train)

    #Predict on test set
    predictions2 = model2.predict(X_test_vectorized)

    #score
    roc2 = roc_auc_score(y_test, predictions2)
    accuracy2 = accuracy_score(y_test, predictions2)
    precision2 = precision_score(y_test, predictions2)
    recall2 = recall_score(y_test, predictions2)
    f1_score2 = f1_score(y_test, predictions2)
   
    #plt.title('Receiver Operating Characteristic')
    #plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.3f'% roc_auc)
    #plt.legend(loc='lower right')
    #plt.plot([0,1],[0,1],'r--')
    #plt.ylabel('True Positive Rate')
    #plt.xlabel('False Positive Rate')

#Tf-Idf

#min_df is used for removing terms that appear too infrequently. For example:
#min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
#min_df = 5 means "ignore terms that appear in less than 5 documents".

#max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:
#max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
#max_df = 25 means "ignore terms that appear in more than 25 documents".

    vect_tf = TfidfVectorizer(min_df= min_df, max_df= max_df).fit(X_train)
    #check2 = vect_tf.get_feature_names()
    
    #transform
    X_train_vectorized_tf = vect_tf.transform(X_train)
    
    #logistic regression
    model1.fit(X_train_vectorized_tf, y_train)

    #Predict on test set
    predictions3 = model1.predict(vect_tf.transform(X_test))

    #score
    roc3 = roc_auc_score(y_test, predictions3)
    accuracy3 = accuracy_score(y_test, predictions3)
    precision3 = precision_score(y_test, predictions3)
    recall3 = recall_score(y_test, predictions3)
    f1_score3 = f1_score(y_test, predictions3)
    
    #svm 
    model2.fit(X_train_vectorized_tf, y_train)

    #Predict on test set
    predictions4 = model2.predict(vect_tf.transform(X_test))

    #score
    roc4 = roc_auc_score(y_test, predictions4)
    accuracy4 = accuracy_score(y_test, predictions4)
    precision4 = precision_score(y_test, predictions4)
    recall4 = recall_score(y_test, predictions4)
    f1_score4 = f1_score(y_test, predictions4)
    
#ngrams
# document frequency of 5 and extracting 1-grams and 2-grams
    vect3 = CountVectorizer(min_df=min_df, ngram_range=(ngram_range1, ngram_range2)).fit(X_train)
    
    X_train_vectorized_ng = vect3.transform(X_train)

    #logistic regression
    model1.fit(X_train_vectorized_ng, y_train)

    #Predict on test set
    predictions5 = model1.predict(vect3.transform(X_test))

    #score
    roc5 = roc_auc_score(y_test, predictions5)
    accuracy5 = accuracy_score(y_test, predictions5)
    precision5 = precision_score(y_test, predictions5)
    recall5 = recall_score(y_test, predictions5)
    f1_score5 = f1_score(y_test, predictions5)
    
    #svm 
    model2.fit(X_train_vectorized_ng, y_train)

    #Predict on test set
    predictions6 = model2.predict(vect3.transform(X_test))

    #score
    roc6 = roc_auc_score(y_test, predictions6)
    accuracy6 = accuracy_score(y_test, predictions6)
    precision6 = precision_score(y_test, predictions6)
    recall6 = recall_score(y_test, predictions6)
    f1_score6 = f1_score(y_test, predictions6)
 
    output = {
         'Accuracy, bow LR:': accuracy1,  
         'ROC, bow LR:': roc1,    
         'Precision, bow LR:': precision1,    
         'Recall, bow LR:': recall1,    
         'F1-score, bow LR:': f1_score1,    
         
         'Accuracy, bow SVM:': accuracy2,  
         'ROC, bow SVM:': roc2,    
         'Precision, bow SVM:': precision2,    
         'Recall, bow SVM:': recall2,    
         'F1-score, bow SVM:': f1_score2,  
         
         'Accuracy, tfidf LR:': accuracy3,  
         'ROC, tfidf LR:': roc3,    
         'Precision, tfidf LR:': precision3,    
         'Recall, tfidf LR:': recall3,    
         'F1-score, tfidf LR:': f1_score3,    
         
         'Accuracy, tfidf SVM:': accuracy4,  
         'ROC, tfidf SVM:': roc4,    
         'Precision, tfidf SVM:': precision4,    
         'Recall, tfidf SVM:': recall4,    
         'F1-score, tfidf SVM:': f1_score4,  
         
         'Accuracy, with ngrams LR:': accuracy5,  
         'ROC, with ngrams LR:': roc5,    
         'Precision, with ngrams LR:': precision5,    
         'Recall, with ngrams LR:': recall5,    
         'F1-score, with ngrams LR:': f1_score5,    
         
         'Accuracy, with ngrams SVM:': accuracy6,  
         'ROC, with ngrams SVM:': roc6,    
         'Precision, with ngrams SVM:': precision6,    
         'Recall, with ngrams SVM:': recall6,    
         'F1-score, with ngrams SVM:': f1_score6  
	}
    
    return output

#%%
   
output1 = tfidf_so_on(rock, hip_hop, 'Lyrics', 5, 0.9, 1, 3);

hiphop1 = hip_hop[hip_hop['Year'] < 2010]
hiphop2 = hip_hop[hip_hop['Year'] >= 2010]

output2 = tfidf_so_on(hiphop1, hiphop2, 'Lyrics', 5, 0.9, 1, 3);

#%%
#Doc2Vec 
data11 = hip_hop.copy()
data22 = rock.copy()
text = 'Lyrics'

data11 = pd.DataFrame(data11[text])
data11['Target'] = 1

data22 = pd.DataFrame(data22[text])
data22['Target'] = 0

min1 = min(len(data11), len(data22));
data11 = data11.sample(n=min1, random_state=1905)
data22 = data22.sample(n=min1, random_state=1905)
data_all2 = data11.append(data22, ignore_index=True)

train, test = train_test_split(data_all2, test_size=0.3, random_state=42)

from nltk.corpus import stopwords

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

train_tagged = train.apply(lambda r: TaggedDocument(words=tokenize_text(r[text]), tags=[r.Target]), axis=1)
test_tagged = test.apply(lambda r: TaggedDocument(words=tokenize_text(r[text]), tags=[r.Target]), axis=1)

#train_tagged.values[12]

#DBOW is the Doc2Vec model analogous to Skip-gram model in Word2Vec. The paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.
#We set the minimum word count to 2 in order to discard words with very few occurrences.

cores = multiprocessing.cpu_count()

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)


logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha
    
y_train, X_train = vec_for_learning(model_dmm, train_tagged)
y_test, X_test = vec_for_learning(model_dmm, test_tagged)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    
new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
