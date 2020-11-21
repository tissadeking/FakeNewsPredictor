#Import the libraries
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

import pickle
#Import the dataset
data = pd.read_csv('news.csv')
#Get the shape of the dataset
print(data.shape)
#Assign the label fake or real as variable y and text as variable X
X = data['text']
y = data['label']
#Split the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 6)

#PASSIVE AGGRESSIVE CLASSIFIER
#Create a pipeline of Tfidf Vectorizer and Passive Aggressive Classifier
pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                    ('pamodel', PassiveAggressiveClassifier())])
#Train the model with the train data
pipeline.fit(X_train, y_train)
#Predict the label of the test data
y_pred = pipeline.predict(X_test)
#Evaluate the performance of the model
score = metrics.accuracy_score(y_test, y_pred)
print('Passive Aggressive Classifier')
print("accuracy:   %0.3f" % score)
print(confusion_matrix(y_test, y_pred))
#Row 1, column 1 of confusion matrix is True Positive ie Accurate Fake
#Row 1, column 2 of confusion matrix is False Negative ie Mistake Real
#Row 2, column 1 of confusion matrix is False Positive ie Mistake Fake
#Row 2, column 2 of confusion matrix is True Negative ie Accurate Real
print(classification_report(y_test, y_pred))

#LOGISTIC REGRESSION MODEL
pipeline2 = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                    ('logreg', LogisticRegression())])
#Train the model with the train data
pipeline2.fit(X_train, y_train)
#Predict the label of the test data
y_pred2 = pipeline2.predict(X_test)
#Evaluate the performance of the model
score2 = metrics.accuracy_score(y_test, y_pred2)
print('Logistic Regression')
print("accuracy:   %0.3f" % score2)
print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))

#NB MULTIMONIAL MODEL
pipeline3 = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                    ('Mnb', MultinomialNB())])
#Train the model with the train data
pipeline3.fit(X_train, y_train)
#Predict the label of the test data
y_pred3 = pipeline3.predict(X_test)
#Evaluate the performance of the model
score3 = metrics.accuracy_score(y_test, y_pred3)
print('NB Multimonial')
print("accuracy:   %0.3f" % score3)
print(confusion_matrix(y_test, y_pred3))
print(classification_report(y_test, y_pred3))


#Serialise the file
with open('model.pickle', 'wb') as handle:
    pickle.dump(pipeline3, handle, protocol=pickle.HIGHEST_PROTOCOL)