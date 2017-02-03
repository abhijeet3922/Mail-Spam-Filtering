# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:00:44 2017

@author: Abhijeet Singh
"""

import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC


def make_Dictionary(root_dir):
    emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]    
    all_words = []       
    for emails_dir in emails_dirs:
        dirs = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
        for d in dirs:
            emails = [os.path.join(d,f) for f in os.listdir(d)]
            for mail in emails:
                with open(mail) as m:
                    for line in m:
                        words = line.split()
                        all_words += words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    
    np.save('dict_enron.npy',dictionary) 
    
    return dictionary
    
def extract_features(root_dir): 
    emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]  
    docID = 0
    features_matrix = np.zeros((33716,3000))
    train_labels = np.zeros(33716)
    for emails_dir in emails_dirs:
        dirs = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
        for d in dirs:
            emails = [os.path.join(d,f) for f in os.listdir(d)]
            for mail in emails:
                with open(mail) as m:
                    all_words = []
                    for line in m:
                        words = line.split()
                        all_words += words
                    for word in all_words:
                      wordID = 0
                      for i,d in enumerate(dictionary):
                        if d[0] == word:
                          wordID = i
                          features_matrix[docID,wordID] = all_words.count(word)
                train_labels[docID] = int(mail.split(".")[-2] == 'spam')
                docID = docID + 1                
    return features_matrix,train_labels
    
#Create a dictionary of words with its frequency

root_dir = 'Enron-data-set'
dictionary = make_Dictionary(root_dir)


#Prepare feature vectors per training mail and its labels

features_matrix,labels = extract_features(root_dir)
np.save('enron_features_matrix.npy',features_matrix)
np.save('enron_labels.npy',labels)


#train_matrix = np.load('enron_features_matrix.npy');
#labels = np.load('enron_labels.npy');
print features_matrix.shape
print labels.shape
print sum(labels==0),sum(labels==1)
X_train, X_test, y_train, y_test = train_test_split(features_matrix, labels, test_size=0.40)

## Training models and its variants

model1 = LinearSVC()
model2 = MultinomialNB()

model1.fit(X_train,y_train)
model2.fit(X_train,y_train)

result1 = model1.predict(X_test)
result2 = model2.predict(X_test)

print confusion_matrix(y_test, result1)
print confusion_matrix(y_test, result2)

