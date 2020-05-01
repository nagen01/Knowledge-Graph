import os
import re
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


def getDataFromFile(filename):
    with open(filename, mode="r",encoding="utf-8") as f:
        news = f.read()
    
    return news

#print(getDataFromFile("/home/nb01/C_Drive/KG/KG_Integration/news/2.txt"))
    

# Check if a word is a Stopword
def isNotStopWord(word):
    return word not in stopwords.words('english')

#Main preprocessing of web scrapped data
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

def getEntityList():
    entity_list = []
    return entity_list

def getDocumentObject(text):
    return nlp(text)


def getArticleUnicodedMentions(doc):
    mentions = []
    if doc._.has_coref:
        for coref_cluster in doc._.coref_clusters:
            mentions.append(coref_cluster.main)
            coref_resolved_doc = doc._.coref_resolved        
        return mentions, coref_resolved_doc
    else:
        #doc = getDocumentObject(doc)
        return mentions, doc


def getEntitySents(entity, coref_resolved_doc):
    sents = ""
    sents_flag = False
    for sent in nltk.sent_tokenize(coref_resolved_doc):
        #print(sent)
        #print("\n")
        entity_found = False
        sent = preprocessingText(sent)
        sent_doc = getDocumentObject(sent)
        for ent in sent_doc.ents:
            print(ent)
            if entity == str(ent):
                entity_found = True
                
        if entity_found:
            sents = sents + sent
            sents_flag = True
    #print(sents)
    return sents, sents_flag


def getSentimentCustom(sents):
    #Loading the sentiment analysis model & vctorizer and working on sentiment analysis of Twitter tweets:
    with open('/home/nb01/C_Drive/KG/KG_Integration/Sentiment analysis/sentiment_analysis_model.pickle','rb') as f:
        clf = pickle.load(f)
    
    with open('/home/nb01/C_Drive/KG/KG_Integration/Sentiment analysis/tfidf_vect.pickle','rb') as f:
        vect = pickle.load(f)
        
    sentiment = clf.predict(vect.transform([sents]).toarray())
    #print("Sentiment score wrt entity =  ",sentiment[0])
    
    if sentiment[0] == 1:
        return "Positive"
    else:
        return "Negative"

#def getSentimentAllenNLP(sents):
#    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.02.17.tar.gz")
#    
#    return predictor.predict(sentence=article)
        

#def process(article, entity, wikidata_url):
def process(entity, filename): 
    article = getDataFromFile(filename)
    #print(article)
    doc = getDocumentObject(article)
    #print(doc)
    #print(entity)
    mentions, coref_resolved_doc = getArticleUnicodedMentions(doc)
    print(coref_resolved_doc)
    print(type(coref_resolved_doc))
    sents, sents_flag = getEntitySents(entity, coref_resolved_doc)
    #print(sents)
    #print(sents_flag)
    if sents_flag:
        #sentiment = getSentimentAllenNLP(sents)
        sentiment = getSentimentCustom(sents)
    else:
        sentiment = "Neutral"
    #print(sents)
    #print(mentions)
    #print(sentiment)
    
    return entity, sentiment

news_loc = '/home/nb01/C_Drive/KG/KG_Integration/news/'
news_entries = os.listdir(news_loc)
print(news_entries)  
#entity_list = getEntityList()
entity_url_list = [["Apple","Q312"], ["Foxconn","Q313"], ["China","Q314"]]
for entity_url in entity_url_list:
    print(entity_url)
    print(entity_url[0])
    for entry in news_entries:
        print(entry)
        filename = news_loc + entry
        print(filename)
        entity, sentiment = process(entity_url[0], filename)
        print(f"For {entity} with wikidata url id {entity_url[1]}, given article has {sentiment} sentiment")
        