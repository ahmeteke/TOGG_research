import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
%matplotlib inline

df = pd.read_csv("./TOGG_tweets_data.csv", dtype={'id': str}) 

from utils import process_wc
wc_text = df.ceviri.apply(lambda w: process_wc(w))
STOPWORDS.add('togg') # togg is a very frequency word so it is cleaned from tweets

wordcloud = WordCloud(stopwords=STOPWORDS, max_words=100, background_color="white", \
                     min_font_size=5, random_state=0).generate(text)
plt.figure(figsize=(16,14))
plt.imshow(wordcloud, interpolation='bilinear') #wordcloud.recolor(color_func=grey_color_func, random_state=3)
plt.axis("off")
plt.show()
