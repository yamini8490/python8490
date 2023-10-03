# %load Sentiment Analysis.py
import pandas as pd

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
#nltk.download('vader_lexicon')

from nltk.stem import WordNetLemmatizer
sia= SentimentIntensityAnalyzer()
df = pd.read_csv('feedback.csv')
print(df)
df['feedback'].values[1]
df['scores']=df['feedback'].apply(lambda body: sia.polarity_scores(str(body)))
df.head()
df['pos']=df['scores'].apply(lambda pos_dict:pos_dict['pos'])
df.head()
df['neg']=df['scores'].apply(lambda neg_dict:neg_dict['neg'])
df['neu']=df['scores'].apply(lambda neg_dict:neg_dict['neu'])
df.head()

df.to_csv('feedback.csv')
