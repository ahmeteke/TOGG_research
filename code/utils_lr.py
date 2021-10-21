import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer 
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words('english')
    # adding the most often and meaningless words for sentiment analysis to stopword list
    stopwords_english.extend(['togg', 'toggs', 'togger', 'also', 'teacher'])
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'[0-9]+', '', tweet) #rakamları temizle
    
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    
    tweet = re.sub(r'ötv', 'tax', tweet)
    
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    
    #tweet_tokens = [w.lower() for w in tweet.split()]
    
    #print(tweet_tokens)
    
    tweets_clean = []
    for word in tweet_tokens:
        if word in ['🇹', '🇷', 'th']:
            continue
        if (word not in stopwords_english and word not in string.punctuation): # remove punctuation and stopwords
            # tweets_clean.append(word)
            lemma_word = lemmatizer.lemmatize(word)  # stemming word
            if lemma_word == 'domesticlithium':
                tweets_clean.append('domestic')
                tweets_clean.append('lithium')
                
                continue
            tweets_clean.append(lemma_word)
    return tweets_clean
