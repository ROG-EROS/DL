# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 18:49:13 2023

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#************************************************************************
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier as knn 
from sklearn.linear_model import LogisticRegression as lrc
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Conv1D,GlobalMaxPooling1D,Dense,Dropout
#************************************************************************
import nltk
nltk.download('punkt')  # Run atleast once  with mobile net not with JIO
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger') # Run atleast once 
from nltk import FreqDist
#************************************************************************

print("********************************************************************************************************************")
print("Load dataset")
print("********************************************************************************************************************")
data = pd.read_csv('spam.csv')
Input_col_name = 'text'
Output_col_name = 'label'
positive_class_label = 1




print("********************************************************************************************************************")
print("Data Analysis")
print("********************************************************************************************************************")
print("Data Head  is : \n",data.head(3))
print("------------------------------------------------------------------------------------")
print("Column names are : \n",data.columns)
print("------------------------------------------------------------------------------------")
print("Data Shape is : \n",data.shape)
print("------------------------------------------------------------------------------------")
print("Data Desc  is : \n",data.describe())
print("------------------------------------------------------------------------------------")
print("Classes info  : \n\t\t\t Sentiment\tCount \n", data.groupby(Output_col_name).size())
print("------------------------------------------------------------------------------------")


print("********************************************************************************************************************")
print("Data filtering - 1. Remove punctuations")
print("********************************************************************************************************************")
data['review_processed'] = data[Input_col_name].str.replace("[^a-zA-Z#]", " ") 
data = data[[Input_col_name,'review_processed',Output_col_name]]
print("Data  head is : \n",data.drop(Input_col_name,axis =1).head(3))
print("------------------------------------------------------------------------------------")

print("********************************************************************************************************************")
print("Data filtering - 2. Remove short words")
print("********************************************************************************************************************")
data['review_processed'] = data['review_processed'].apply(lambda x: ' '.join([w for w in str(x).split() if len(w)>2]))
print("Data  head is : \n",data.drop(Input_col_name,axis =1).head(3))
print("------------------------------------------------------------------------------------")

print("********************************************************************************************************************")
print("Data filtering - 3. Convert to lower case")
print("********************************************************************************************************************")
data['review_processed'] = [review.lower() for review in data['review_processed']]
print("Data  head is : \n",data.drop(Input_col_name,axis =1).head(3))
print("------------------------------------------------------------------------------------")

print("********************************************************************************************************************")
print("Data filtering - 4. Remove stop words")
print("********************************************************************************************************************")
stop_words = stopwords.words('english')
#************************************************************************
add_words = []#'product','flipkart','good','money','delivery', 'also']
#************************************************************************
stop_words.extend(add_words)
#************************************************************************
def remove_stopwords(rev):
    review_tokenized = word_tokenize(rev)
    rev_new = " ".join([i for i in review_tokenized  if i not in stop_words])
    return rev_new
#************************************************************************
data['review_processed'] = [remove_stopwords(r) for r in data['review_processed']]
print("Data  head is : \n",data.drop(Input_col_name,axis =1).head(3))
print("------------------------------------------------------------------------------------")
#************************************************************************


print("********************************************************************************************************************")
print("Data filtering - 5. Lemmatization")
print("********************************************************************************************************************") 
lemmatizer = WordNetLemmatizer()
#************************************************************************
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
#************************************************************************
def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)
#************************************************************************
#data['review_processed'] = data['review_processed'].apply(lambda x: lemmatize_sentence(x))
#************************************************************************


print("********************************************************************************************************************")
print("Plotting most frequent words")
print("********************************************************************************************************************") 
sns.set(style = 'white')
#************************************************************************
all_words_data = data[data[Output_col_name] == positive_class_label]
#************************************************************************
all_words = ' '.join([text for text in all_words_data ['review_processed']])
all_words = all_words.split()
words_data = FreqDist(all_words)
#************************************************************************
words_data = pd.DataFrame({'word':list(words_data.keys()), 'count':list(words_data.values())})
words_data = words_data.nlargest(columns="count", n = 20) 
words_data.sort_values('count', inplace = True)
#************************************************************************
plt.figure(figsize=(20,5))
ax = plt.barh(words_data['word'], width = words_data['count'])
#ax.set_ylabel('Count')
plt.show()
#************************************************************************



print("********************************************************************************************************************")
print("Bilding a Word Cloud")
print("********************************************************************************************************************") 

word_cloud_data = data[data[Output_col_name] == positive_class_label]
all_words = ' '.join([text for text in word_cloud_data['review_processed']])
 

wordcloud = WordCloud(width = 800, height = 800, 
                      background_color ='white', 
                      min_font_size = 10).generate(all_words)
#************************************************************************                
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()#block = False)
#plt.figure()
#************************************************************************


print("********************************************************************************************************************")
print("Apply Count Vectorizer & Prepare datasets")
print("********************************************************************************************************************") 
cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(data.review_processed).toarray()
y = data[Output_col_name]
#************************************************************************


print("********************************************************************************************************************")
print("Prepare datasets for both trainig and testing")
print("********************************************************************************************************************")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


print("********************************************************************************************************************")
print("Init model")
print("********************************************************************************************************************")
model = MultinomialNB()
model=Sequential()
model.add(Embedding(5000,100,input_length=100))
model.add(Conv1D(64,5,activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
print("********************************************************************************************************************")
print("Train model")
print("********************************************************************************************************************")
model.fit(X_train, y_train)


print("********************************************************************************************************************")
print("Model validation")
print("********************************************************************************************************************")
predicted = model.predict(X_test)


print("********************************************************************************************************************")
print("Validate using metrics")
print("********************************************************************************************************************")
accuracyScore = round(accuracy_score(predicted, y_test) * 100,2)
print("Accuracuy Score: \n",accuracyScore )
print("------------------------------------------------------------------------------------")
c_matrix = confusion_matrix(predicted, y_test)
print("Confusion matrix: \n",c_matrix )
print("------------------------------------------------------------------------------------")
plt.figure(figsize = (8, 10), facecolor = None) 
group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in c_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in c_matrix.flatten()/np.sum(c_matrix)]
labels = [f"{v2}\n{v3}" for v2, v3 in zip( group_counts, group_percentages)]
data_classes_cnt= len(set(data[Output_col_name]))
labels = np.asarray(labels).reshape(data_classes_cnt,data_classes_cnt)
sns.heatmap(c_matrix, annot=labels, fmt="", cmap='Blues')
plt.show()
print("------------------------------------------------------------------------------------")

print("********************************************************************************************************************")
print("Program ends here")
print("********************************************************************************************************************")
