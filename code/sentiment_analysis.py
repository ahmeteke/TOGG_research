import pandas as pd
import numpy as np
import re
import os
from utils_lr import process_tweet

tweet = pd.read_csv("../data/labeled_data.csv", header=0)

import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

porter = PorterStemmer()
lemma = WordNetLemmatizer()

def tweet_tokenize(text):
    return process_tweet(text)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tweet.translated, 
                                                    tweet.label.astype('int'), 
                                                    random_state=0, 
                                                    test_size = 0.2)

cv = CountVectorizer(tokenizer=tweet_tokenize, analyzer='word', min_df=3)
cv_data = cv.fit_transform(X_train)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#
### Logistic regression model set up with python sklearn libary 
#
from sklearn.linear_model import LogisticRegression

modelLog = LogisticRegression()
modelLog.fit(cv_data, y_train)
predict = modelLog.predict(cv.transform(X_test))
print(predict.mean()) # 0.83
print("Test set Evaluation Results for Logistic Regression: ")
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tahminLog))) # 0.80
print('Precision: {:.2f}'.format(precision_score(y_test, tahminLog))) # 0.84
print('Recall: {:.2f}'.format(recall_score(y_test, tahminLog))) # 0.89
print('F1: {:.2f}'.format(f1_score(y_test, tahminLog))) # 0.87
