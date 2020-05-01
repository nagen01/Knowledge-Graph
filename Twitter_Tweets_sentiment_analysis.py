# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 09:00:54 2019

@author: Nagendra
"""
#Importing important libraries
import pickle
import re
import tweepy
from tweepy import OAuthHandler
import numpy as np
import matplotlib.pyplot as plt

#Loading the sentiment analysis model & vctorizer and working on sentiment analysis of Twitter tweets:
with open('sentiment_analysis_model.pickle','rb') as f:
    clf = pickle.load(f)

with open('tfidf_vect.pickle','rb') as f:
    vect = pickle.load(f)

#Initializing the keys:
consumer_key = "tAcWjBz3QNk9hw3qGsPcfKqjr"
consumer_secret = "iPbpnTT3YYxretX580SgpOZtBXDZtfBA9ZVsuQk917Ekrp0cRF"
access_token = "854563084693680128-DqGD5n6W18DidZIfAGZDo5Zpxxv3CWZ"
access_secret = "9vDIO0ZWIr6gtfVElGV9DJwTTopP19CeHh0Xzz8QrGydi"

#Providing the authorization:
auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)
api = tweepy.API(auth,timeout=10)

#Fetching real time tweets from Twitter app/account
args = ['facebook']

list_tweets = []

query = args[0]
if len(args) == 1:
    for status in tweepy.Cursor(api.search,q=query+' -filter:retweets',lang='en',result_type='recent').items(1000):
        list_tweets.append(status.text)

total_pos = 0
total_neg = 0
#Preprocessing the tweets
for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s+"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s+"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$"," ",tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"aren't","are not",tweet)
    tweet = re.sub(r"isn't","is not",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"don't","do not",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"we're","we are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    sentiment = clf.predict(vect.transform([tweet]).toarray())
    print(tweet, " ", sentiment)
    
    if sentiment[0] == 1:
        total_pos += 1
    else:
        total_neg += 1
        
#Ploting this on histogram:
objects = ['Positive','Negative']
y_pos = np.arange(len(objects))

plt.bar(y_pos,[total_pos,total_neg],alpha=0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Number')
plt.title('Number of Positive and Negative Tweets')

plt.show()