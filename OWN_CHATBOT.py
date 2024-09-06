# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:48:35 2023
-------------------
Created by nagal
-------------------
@author: nagal
"""
print("********************************************************************************************************************")
print("Program starts here")
print("********************************************************************************************************************")


print("********************************************************************************************************************")
print("Import libraries")
print("********************************************************************************************************************")
import warnings
warnings.filterwarnings("ignore")
import nltk
import numpy as np
import random
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only


print("********************************************************************************************************************")
print("Load data file")
print("********************************************************************************************************************")
f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()


print("********************************************************************************************************************")
print("Pre-processing the raw text")
print("********************************************************************************************************************")
# converts to lowercase
raw=raw.lower()
# converts to list of sentences 
sent_tokens = nltk.sent_tokenize(raw)
# converts to list of words
word_tokens = nltk.word_tokenize(raw)
lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
#Keyword matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        

print("********************************************************************************************************************")
print("Analyze the data")
print("********************************************************************************************************************")       
print(sent_tokens[:2])
print(word_tokens[:2])


print("********************************************************************************************************************")
print("Generating Response")
print("********************************************************************************************************************") 
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

flag=True
print("ROBO: My name is Robo, your Personal Agent. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                print("ROBO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")
        
print("********************************************************************************************************************")
print("Program ends here")
print("********************************************************************************************************************")