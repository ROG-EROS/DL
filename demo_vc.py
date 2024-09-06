# Import CV
from sklearn.feature_extraction.text import CountVectorizer
 
#create doc_text
doc_text = ["teachers  teache many students", 
            "Many engineers work for society", 
            "teachers and students build society"]
 
# Vectorizer Object
vectorizer = CountVectorizer()
vectorizer.fit(doc_text)
 
# Encode the doc_text
vector = vectorizer.transform(doc_text)

# count words
text_counts = vectorizer.fit_transform(doc_text)

# Printing the identified Unique words along with their indices
print("Vocabulary: ", vectorizer.vocabulary_)

# Summarizing the Encoded Texts
print("Encoded doc_text is:")
print(vector.toarray())
print(text_counts)