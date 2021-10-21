import pandas as pd
import numpy as np
import nltk
import re
import os
from utils_to import process_tweet

topic = pd.read_csv('../data/TOGG_tweets_data.csv', header=0)

# lithium typo fix using regular expression
topic.translated = topic.translated.str.replace(r'\S*(L|l)ithium', 'lithium')
topic.translated = topic.translated.str.replace(r'\S*(L|l)ityum', 'lithium')
# engine instead of motor 
topic.translated = topic.translated.str.replace(r'\S*motor', 'engine')

# Python Gensim library was used to generate LDA model for topic modeling analysis
import gensim
from gensim.utils import simple_preprocess

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=False))  # deacc=True removes punctuations
