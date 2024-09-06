# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:42:28 2023
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#************************************************************************
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#************************************************************************
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
#************************************************************************
#import pickle5 as pickle
#************************************************************************


print("********************************************************************************************************************")
print("Load dataset")
print("********************************************************************************************************************")
data = pd.read_csv('dreaddit.csv')
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
print("Data processing")
print("********************************************************************************************************************")
# Preprocess the dataset
data[Output_col_name] = data[Output_col_name].apply(lambda x: 'hate' if x >0 else 'normal')
data = data[[Input_col_name, Output_col_name]]
data = data.sample(frac=1).reset_index(drop=True)
#************************************************************************
# Tokenize and pad the review sequences
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(data[Input_col_name])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(data[Input_col_name])
padded_sequences = pad_sequences(sequences, maxlen=100, truncating='post')
#************************************************************************
# Convert the sentiment labels to one-hot encoding
sentiment_labels = pd.get_dummies(data[Output_col_name]).values
#************************************************************************



print("********************************************************************************************************************")
print("Prepare datasets for both trainig and testing")
print("********************************************************************************************************************")
x_train, x_test, y_train, y_test = train_test_split(padded_sequences, sentiment_labels, test_size=0.2)


print("********************************************************************************************************************")
print("Init model")
print("********************************************************************************************************************")
model = Sequential()
#************************************************************************
model.add(Embedding(5000, 100, input_length=100))
model.add(Conv1D(64, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))
#************************************************************************
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#************************************************************************
model.summary()
#************************************************************************


print("********************************************************************************************************************")
print("Train model")
print("********************************************************************************************************************")
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

print("\n********************************************************************")
print("\t\t\t Plot Acc & Loss ..")
print("The following metrics data available in history -")
for key in history.history.keys():
    print(key)
fig = plt.figure(figsize=(16, 12))
# Plot Accuracy levels during traing
fig.add_subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Validation'], loc='lower left')
plt.ylabel('Accuracy')
plt.xlabel('Epoch Number')
plt.title('Model accuracy during training')
plt.show()
# Plot loss levels during traing
fig.add_subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Validation'], loc='upper left')
plt.ylabel('Loss')
plt.xlabel('Epoch Number')
plt.title('Model loss during training')
plt.show()
print("********************************************************************")


print("********************************************************************************************************************")
print("Model validation")
print("********************************************************************************************************************")
y_pred = np.argmax(model.predict(x_test), axis=-1)
print("Accuracy:", accuracy_score(np.argmax(y_test, axis=-1), y_pred))


'''
print("********************************************************************************************************************")
print("Save Model for future")
print("********************************************************************************************************************")
model.save('my_first_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    


print("********************************************************************************************************************")
print("Load previously trained model")
print("********************************************************************************************************************")
model = keras.models.load_model('my_first_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
'''
print("********************************************************************************************************************")
print("pre-process test / inference data")
print("********************************************************************************************************************")
def predict_sentiment(text):
    # Tokenize and pad the input text
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence = pad_sequences(text_sequence, maxlen=100)

    # Make a prediction using the trained model
    predicted_rating = model.predict(text_sequence)[0]
    if np.argmax(predicted_rating) == 0:
        return 'Negative'
    elif np.argmax(predicted_rating) == 1:
        return 'Neutral'
    else:
        return 'Positive'
    

print("********************************************************************************************************************")
print("Test / inference ")
print("********************************************************************************************************************")
test_data = ["I  loved my stay at that hotel. Room was fantastic!",
             "I hate that product. Will not buy it again",
             "Overall, it was an average experience"]

for i in range(len(test_data)):
    predicted_sentiment = predict_sentiment(test_data[i])
    #print(predicted_sentiment)



