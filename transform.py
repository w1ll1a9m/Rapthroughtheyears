#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:46:45 2019

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

xx = pd.read_csv('hip_hop_nocontracted_v4_emotions.csv')

dummy = pd.get_dummies(xx, columns=["Sentiment"])

dummy.to_csv("sentiment123.csv")