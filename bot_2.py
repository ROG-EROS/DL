# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:18:12 2023

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
import json
import string
import random
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer # It has the ability to lemmatize.
import tensorflow as tensorF # A multidimensional array of elements is represented by this symbol.
from tensorflow.keras import Sequential # Sequential groups a linear stack of layers into a tf.keras.Model
from tensorflow.keras.layers import Dense, Dropout

#nltk.download("punkt")# required package for tokenization
#nltk.download("wordnet")# word database


print("********************************************************************************************************************")
print("Create your own rules/ data / situation")
print("********************************************************************************************************************")
data = {"intents": [

             {"tag": "age",
              "patterns": ["how old are you?"],
              "responses": ["I am 2 years old and my birthday was yesterday"]
             },
              {"tag": "greeting",
              "patterns": [ "Hi", "Hello", "Hey"],
              "responses": ["Hi there", "Hello", "Hi :)"],
             },
              {"tag": "goodbye",
              "patterns": [ "bye", "later"],
              "responses": ["Bye", "take care"]
             },
             {"tag": "name",
              "patterns": ["what's your name?", "who are you?"],
              "responses": ["I have no name yet," "You can give me one, and I will appreciate it"]
             }]}


print("********************************************************************************************************************")
print("process data")
print("********************************************************************************************************************")
lm = WordNetLemmatizer() #for getting words
# lists
cls_data = []
newWords = []
documentX = []
documentY = []
# Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        ournewTkns = nltk.word_tokenize(pattern)# tokenize the patterns
        newWords.extend(ournewTkns)# extends the tokens
        documentX.append(pattern)
        documentY.append(intent["tag"])


    if intent["tag"] not in cls_data:# add unexisting tags to their respective classes
        cls_data.append(intent["tag"])

newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation] # set words to lowercase if not in punctuation
newWords = sorted(set(newWords))# sorting words
cls_data = sorted(set(cls_data))# sorting classes


print("********************************************************************************************************************")
print("Create datasets")
print("********************************************************************************************************************")
trainingData = [] # training list array
outEmpty = [0] * len(cls_data)
# bow model
for idx, doc in enumerate(documentX):
    bagOfwords = []
    text = lm.lemmatize(doc.lower())
    for word in newWords:
        bagOfwords.append(1) if word in text else bagOfwords.append(0)

    outputRow = list(outEmpty)
    outputRow[cls_data.index(documentY[idx])] = 1
    trainingData.append([bagOfwords, outputRow])

random.shuffle(trainingData)
trainingData = num.array(trainingData, dtype=object)# coverting our data into an array afterv shuffling

x = num.array(list(trainingData[:, 0]))# first trainig phase
y = num.array(list(trainingData[:, 1]))# second training phase


iShape = (len(x[0]),)
oShape = len(y[0])

print("********************************************************************************************************************")
print("Init model")
print("********************************************************************************************************************")
model = Sequential()


print("********************************************************************************************************************")
print("add layers to model")
print("********************************************************************************************************************")
model.add(Dense(128, input_shape=iShape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(oShape, activation = "softmax"))


print("********************************************************************************************************************")
print("Compile options")
print("********************************************************************************************************************")
md = tensorF.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=md,
              metrics=["accuracy"])
print(model.summary())


print("********************************************************************************************************************")
print("train model")
print("********************************************************************************************************************")
model.fit(x, y, epochs=200, verbose=1)




print("********************************************************************************************************************")
print("create methods to process text data")
print("********************************************************************************************************************")
def ourText(text):
  newtkns = nltk.word_tokenize(text)
  newtkns = [lm.lemmatize(word) for word in newtkns]
  return newtkns

def wordBag(text, vocab):
  newtkns = ourText(text)
  bagOwords = [0] * len(vocab)
  for w in newtkns:
    for idx, word in enumerate(vocab):
      if word == w:
        bagOwords[idx] = 1
  return num.array(bagOwords)

def Pclass(text, vocab, labels):
  bagOwords = wordBag(text, vocab)
  res = model.predict(num.array([bagOwords]))[0]
  newThresh = 0.2
  yp = [[idx, res] for idx, res in enumerate(res) if res > newThresh]

  yp.sort(key=lambda x: x[1], reverse=True)
  newList = []
  for r in yp:
    newList.append(labels[r[0]])
  return newList

def getRes(firstlist, fJson):
  tag = firstlist[0]
  listOfIntents = fJson["intents"]
  for i in listOfIntents:
    if i["tag"] == tag:
      res = random.choice(i["responses"])
      break
  return res

print("********************************************************************************************************************")
print("Interact with user")
print("********************************************************************************************************************")
while True:
    newMessage = input(">>> :")
    intents = Pclass(newMessage, newWords, cls_data)
    res = getRes(intents, data)
    print(res)
    print("--------------------------------------------------------------------")
print("********************************************************************************************************************")
print("Program ends here")
print("********************************************************************************************************************")