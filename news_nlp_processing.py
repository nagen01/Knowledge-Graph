import os
import re
import json
import nltk
from nltk.corpus import stopwords
import spacy
import neuralcoref
#from allennlp.predictors.predictor import Predictor
#import allennlp_models.sentiment
import pickle
nlp = spacy.load('en_core_web_sm')
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')        


def isNotStopWord(word):
    return word not in stopwords.words('english')

def preprocessingText(text):
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    sentences = nltk.sent_tokenize(text)
    tokens = []
    temp = ""
    
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        
        #Converting to LowerCase
        #words = map(str.lower, words)
        
        # Remove stop words
        words = filter(lambda x: isNotStopWord(x), words)
        
        # Removing punctuations except '<.>/<?>/<!>'
        punctuations = '"#$%&\'()*+,-/:;<=>@\\^_`{|}~'
        words = map(lambda x: x.translate(str.maketrans('', '', punctuations)), words)
        
        # Remove empty strings
        words = filter(lambda x: len(x) > 0, words)
      
        tokens = tokens + list(words)
        temp = ' '.join(word for word in tokens)
        
    return temp

    
def getPortfilioEntityList(json_portfolio):
    portfolio_entity_list = []
    #portfolio_entity_list = json.loads(str(json_portfolio))
    #print python_obj["name"]
    #print python_obj["city"]
    
    return portfolio_entity_list


def getNewsData(filename):
    with open(filename, mode="r",encoding="utf-8") as f:
        news = f.read()
    
    return news
    
def getDocumentObject(text):
    return nlp(text)

def getArticleEntityList(doc):
    article_entity_list = []
    for ent in doc.ents:
        if str(ent) not in article_entity_list:
            article_entity_list.append(str(ent))
            
    return article_entity_list


def getCorefResolved(doc):
    #if doc._.has_coref:
    #    coref_resolved_doc = doc._.coref_resolved        
    #   return str(coref_resolved_doc)
    #else:
    #    return doc
    return doc


def getEntitySents(entity, coref_resolved_doc):
    sents = ""
    for sent in nltk.sent_tokenize(coref_resolved_doc):
        entity_found = False
        sent_doc = getDocumentObject(sent)
        for ent in sent_doc.ents:
            if entity == str(ent):
                entity_found = True
                
        if entity_found:
            sents = sents + sent
    return sents


def getSentimentCustomModel():
    #Loading the sentiment analysis model & vctorizer and working on sentiment analysis of Twitter tweets:
    with open('/home/nb01/C_Drive/KG/KG_Integration/Sentiment analysis/sentiment_analysis_model.pickle','rb') as f:
        clf = pickle.load(f)
    
    with open('/home/nb01/C_Drive/KG/KG_Integration/Sentiment analysis/tfidf_vect.pickle','rb') as f:
        vect = pickle.load(f)
    
    return clf, vect

def getSentimentCustom(clf, vect, sents):
    sents = preprocessingText(sents)
    sentiment = clf.predict(vect.transform([sents]).toarray())  
    if sentiment[0] == 1:
        return "Positive"
    else:
        return "Negative"
    
        
def compareList(portfolio_entity_list, article_entity_list):
    matching_entities = []
    for e in portfolio_entity_list:
        if e not in article_entity_list:
            continue;
        else:
            matching_entities.append(e)
            
    return matching_entities


def processPortfolioSentiment(portfolio, portfolio_entity_list, filename):
    news_article = getNewsData(filename)
    print("News Article: ", news_article)
    doc = getDocumentObject(news_article)
    article_entity_list = getArticleEntityList(doc)
    print("Entities present in news article: ",article_entity_list)
    matching_entities = compareList(portfolio_entity_list, article_entity_list)
    print("Entities matching with portfolio entities: ", matching_entities)
    
    coref_resolved_doc = getCorefResolved(news_article)
    for entity in matching_entities:
        sents = getEntitySents(entity, coref_resolved_doc)
        clf, vect = getSentimentCustomModel()
        sentiment = getSentimentCustom(clf, vect, sents)
        print(f"For portfolio {portfolio}, it's related entity {entity} has {sentiment} sentiment for given news at {filename}")
        print("\n")           
        

#portfolio, portfolio_entity_list = getPortfilioEntityList(json_portfolio)
portfolio, portfolio_entity_list = "Apple", ["Apple", "Foxconn", "China", "Chinese"]
print("portfolio: ", portfolio)
print("portfolio_entity_list: ", portfolio_entity_list)         
news_loc = '/home/nb01/C_Drive/KG/KG_Integration/news/'
news_filenames = os.listdir(news_loc)
for fname in news_filenames:
    filename = news_loc + fname
    processPortfolioSentiment(portfolio, portfolio_entity_list, filename)