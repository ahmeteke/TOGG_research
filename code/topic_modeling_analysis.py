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

# pre-process using process_tweet module within utils_to.py file
reviews = []
for review in topic.translated:
    text = process_tweet(review)
    reviews.append(text)
    
#data = list(topic_01.translatedText)
data_words = list(sent_to_words(yorumlar))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=3, threshold=100) # higher threshold fewer phrases.

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

# Form Bigrams
data_words_bigrams = make_bigrams(data_words)

import gensim.corpora as corpora

# Create Dictionary
id2word = corpora.Dictionary(data_words_bigrams)

# Create Corpus
texts = data_words_bigrams

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# this is LDA topic model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=4,    # number of topics
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       #alpha=0.01,
                                       #eta=0.01,
                                       per_word_topics=True)

from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)

# Print n most words, n=30
word_dict = {};
for i in range(4):
    words = lda_model.show_topic(i, topn = 30)
    word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words]
pd.DataFrame(word_dict)

# Find dominant topic each tweet and save 
def format_topics_sentences(ldamodel=None, corpus=corpus):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                #wp = ldamodel.show_topic(topic_num)
                #topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4)]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution']

    # Add original text to the end of the output
    #contents = pd.Series(data_text)
    #sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_dominant_topic = format_topics_sentences(ldamodel=lda_model, corpus=corpus)
