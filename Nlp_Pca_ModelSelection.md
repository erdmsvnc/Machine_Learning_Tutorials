# Nlp_Pca_ModelSelection

1-) Natural Language Process
---
```
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
```

2-) Principal Component Analysis
---
```
from sklearn.datasets import load_iris
import pandas as pd

# %%
iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data,columns = feature_names)
df["sinif"] = y

x = data

#%% PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, whiten= True )  # whitten = normalize
pca.fit(x)

x_pca = pca.transform(x)

print("variance ratio: ", pca.explained_variance_ratio_)

print("sum: ",sum(pca.explained_variance_ratio_))

#%% 2D

df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red","green","blue"]

import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.sinif == each],df.p2[df.sinif == each],color = color[each],label = iris.target_names[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
```

3-) Model Selection
---
```
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#%%
iris = load_iris()

x = iris.data
y = iris.target

# %% normalization
x = (x-np.min(x))/(np.max(x)-np.min(x))

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.3)

#%% knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)  # k = n_neighbors

# %% K fold CV K = 10
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn, X = x_train, y= y_train, cv = 10)
print("average accuracy: ",np.mean(accuracies))
print("average std: ",np.std(accuracies))

#%% 
knn.fit(x_train,y_train)
print("test accuracy: ",knn.score(x_test,y_test))


# %% grid search cross validation for knn

from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors":np.arange(1,50)}
knn= KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 10)  # GridSearchCV
knn_cv.fit(x,y)

#%% print hyperparameter KNN algoritmasindaki K degeri
print("tuned hyperparameter K: ",knn_cv.best_params_)
print("tuned parametreye gore en iyi accuracy (best score): ",knn_cv.best_score_)

# %% Grid search CV with logistic regression

x = x[:100,:]
y = y[:100] 

from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv = 10)
logreg_cv.fit(x,y)

print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_)
```
