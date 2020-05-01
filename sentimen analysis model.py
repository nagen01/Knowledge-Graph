# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 22:59:11 2019

@author: Nagendra
"""

#Text classifier : Sentiment analysis

#Importing important libraries
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')
 
#Importing dataset:
reviews = load_files('txt_sentoken')
X,y = reviews.data, reviews.target

#Storing the data as pickle file
with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
#unpickling the dataset:
with open('X.pickle','rb') as f:
    A = pickle.load(f)

with open('y.pickle','rb') as f:
    B = pickle.load(f)
    
#Preprocessing the data and creating Bag of words model
corpus = []
for i in range(len(X)):
    review = re.sub(r'\W',' ',str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review)
    review = re.sub(r'^[a-z]\s+',' ',review)
    review = re.sub(r'\s+',' ',review)
    corpus.append(review)

#Creating bag of words model with the data above
#from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer(max_features=2000,max_df=0.6,min_df=3,stop_words=stopwords.words('english'))
#X = vectorizer.fit_transform(corpus).toarray()

#from sklearn.feature_extraction.text import TfidfTransformer
#transformer = TfidfTransformer()
#X = transformer.fit_transform(X).toarray()

#CountVectorizer and TfidfTransformer can be completed only in one step;
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(max_features=2000,max_df=0.6,min_df=3,stop_words=stopwords.words('english'))
X = tfidf_vect.fit_transform(X).toarray()
#Splitting the data into training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1) 

#Using the logistic regressing to train the model:
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
sentiment_analysis_model = regressor.fit(X_train,y_train)

#testing the model
y_pred = sentiment_analysis_model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Saving the model for future use:
with open('sentiment_analysis_model.pickle','wb') as f:
    pickle.dump(sentiment_analysis_model,f)

with open('tfidf_vect.pickle','wb') as f:
    pickle.dump(tfidf_vect,f)