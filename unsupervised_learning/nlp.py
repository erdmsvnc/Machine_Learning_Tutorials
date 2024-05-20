import pandas as pd


data = pd.read_csv("gender_classifier.csv", encoding="latin1")
data = pd.concat([data.gender, data.description], axis=1)
data.dropna(axis = 0, inplace = True)

data.gender = [1 if each == "female" else 0 for each in data.gender]

#%% cleaning data
#regular expression RE
import re

first_description = data.description[4]

description = re.sub("[^a-zA-Z]"," ",first_description)#a ile z arasındakiler dışındaki karakterleri " " ile doldur

description = description.lower()

#%% stopwords (irrelavent words)gereksiz kelimeler

import nltk #natural language tool kit

from nltk.corpus import stopwords
#splitte "shouldn't gibi kelimeler "should" ve "not" diye ayrılmaz ama word_tokenize da ayrılır

#description = description.split()

description = nltk.word_tokenize(description)
#%%
#gereksiz kelimeleri cikar
description = [word for word in description if not word in set(stopwords.words("english"))]
#%%
#lemmatazation loved => love 

import nltk as nlp


lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word)for word in description]

description = " ".join(description)

#%% 

description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    description = [word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word)for word in description]
    description = " ".join(description)
    description_list.append(description)
    

#%% bag of words

from sklearn.feature_extraction.text import CountVectorizer # bag of words yaratmak icin kullanilir
max_features = 5000 #toplam kelimeleerin icinde en cok kullanilen 5000 kelimeyi al

count_vectorizer = CountVectorizer(max_features=max_features, stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

y = data.iloc[:0].values #male or female classes
x = sparce_matrix
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state= 42)

#%% naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)

#prediction

y_pred = nb.predict(x_test)











    