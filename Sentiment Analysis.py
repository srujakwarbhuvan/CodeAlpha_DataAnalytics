

#%% 📚 Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import nltk
import re

#%% Downloading necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

#%% [Load dataset]
df = pd.read_csv("C:\\tripadvisor_hotel_reviews.csv")  # Adjust path if needed
df.dropna(inplace=True)
df.head()


#%% [Clean text]
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['cleaned'] = df['Review'].apply(clean_text)
df[['Review', 'cleaned']].head()


#%% [Sentiment Analysis using VADER]
vader = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = vader.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment'] = df['cleaned'].apply(get_sentiment)

# 🖨 Preview results
df[['cleaned', 'Sentiment']].head()

#%% [Download required NLTK corpus]
import nltk
nltk.download('punkt')

#%% [Emotion Detection with NRCLex]
from nrclex import NRCLex

def detect_emotions(text):
    emotion = NRCLex(text)
    top_emotion = emotion.top_emotions
    if top_emotion:
        return top_emotion[0][0]
    else:
        return 'neutral'

df['Emotion'] = df['cleaned'].apply(detect_emotions)
df[['cleaned', 'Emotion']].head()


#%% [Emotion Distribution Plot]
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='Emotion', order=df['Emotion'].value_counts().index, palette='Set3')
plt.title("Detected Emotions")
plt.xticks(rotation=45)
plt.show()

#%% [WordCloud for Positive Reviews]
positive_text = ' '.join(df[df['Sentiment'] == 'Positive']['cleaned'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Reviews WordCloud")
plt.show()

# %%
