# study-ML-intro

<img src="https://user-images.githubusercontent.com/31917400/34910969-bf8b8bee-f8b7-11e7-9d23-8d54d974c128.jpg" />

<img src="https://user-images.githubusercontent.com/31917400/35225294-d059a978-ff7f-11e7-9779-b3b51cca7549.jpg" />

---------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------
# project-Python-mini

__Package:__ nltk(natural language toolkit), scikit-learn, numpy, scipy, os, urllib(dataset), tarfile(dataset) 

__Story:__ A couple of years ago, J.K. Rowling (of Harry Potter fame) tried something interesting. She wrote a book, “The Cuckoo’s Calling,” under the name Robert Galbraith. The book received some good reviews, but no one paid much attention to it--until an anonymous tipster on Twitter said it was J.K. Rowling. The London Sunday Times enlisted two experts to compare the linguistic patterns of “Cuckoo” to Rowling’s “The Casual Vacancy,” as well as to books by several other authors. After the results of their analysis pointed strongly toward Rowling as the author, the Times directly asked the publisher if they were the same person, and the publisher confirmed. The book exploded in popularity overnight.

We have a set of emails, half of which were written by one person and the other half by another person at the same company . Our objective is to classify the emails as written by one person or the other based **only on the text of the email.** 

We will start by giving a list of strings. Each string is the text of an email, which has undergone some basic preprocessing; we will then provide the code to split the dataset into training and testing sets.

One particular feature of Naive Bayes is that it’s a good algorithm for working with **text classification.** When dealing with text, it’s very common to treat each unique word as a feature, and since the typical person’s vocabulary is many thousands of words, this makes for a large number of features. The relative simplicity of the algorithm and the independent features assumption of Naive Bayes make it a strong performer for classifying texts. 
 - authors and labels:
   - Sara has label 0
   - Chris has label 1
 - When training you may see the following error: `UserWarning: Duplicate scores. Result may depend on feature ordering.There are probably duplicate features, or you used a classification score for a regression task. warn("Duplicate scores. Result may depend on feature ordering.")` This is a warning that two or more words happen to have the same usage patterns in the emails--as far as the algorithm is concerned, this means that two features are the same. Some algorithms will actually break (mathematically won’t work) or give multiple different answers (depending on feature ordering) when there are duplicate features and sklearn is giving us a warning. Good information, but not something we have to worry about.  
 - If you find that the code is causing memory errors, you can also try setting test_size = 0.5 in the email_preprocess.py file.
```
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
```
 - 'features_train' and 'features_test' are the features for the training and testing datasets, respectively 'labels_train' and 'labels_test' are the corresponding item labels.
```
features_train, features_test, labels_train, labels_test = preprocess()
```


------------------------------------------------------
## 0. DecisionTree
> Titanic: Make predictions about passenger survival
 - Survived: Outcome of survival (0 = No; 1 = Yes)
 - Pclass: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
 - Name: Name of passenger
 - Sex: Sex of the passenger
 - Age: Age of the passenger (Some entries contain NaN)
 - SibSp: Number of siblings and spouses of the passenger aboard
 - Parch: Number of parents and children of the passenger aboard
 - Ticket: Ticket number of the passenger
 - Fare: Fare paid by the passenger
 - Cabin Cabin number of the passenger (Some entries contain NaN)
 - Embarked: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)

Since we're interested in the outcome of survival for each passenger or crew member, we can remove the Survived feature from this dataset and store it as its own separate variable outcomes. We will use these outcomes as our prediction targets.


























## 1. NaiveBayes
Q. Create and train a Naive Bayes classifier. Use it to make predictions for the test set. What is the accuracy?

Q. Compare the time to train the classifier and to make predictions with it. What is faster, training or prediction?
 - accuracy: 0.83
 - training time: 1.729 s
 - predicting time: 0.28 s
```
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

t0 = time() ########################################
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" ###

t0 = time() ##########################################
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s" ###

accuracy = clf.score(features_test, labels_test); accuracy
```

## 2. Support Vector Machine
Q. Tackle the exact same email author ID problem as the above project, but now with an SVM. What we find will help clarify some of the practical differences between the two algorithms. It also gives us a chance to play around with parameters a lot more than Naive Bayes did.
 - accuracy: 0.98407281001137659
 - training time: 397.373 s
 - predicting time: 44.041 s
```
from sklearn.svm import SVC
clf = SVC(kernel='linear')

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" 

t0 = time()  
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test); accuracy 
```
### if too slow  
```
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]
```
These lines effectively slice the training dataset down to 1% of its original size, tossing out 99% of the training data. For example, 
### Voice recognition(Siri) and transaction blocking(ATM) need to happen in real time, with almost no delay, thus in this case, time matters.

Q. What about RBF with different 'C' say, C =10., =100., =1000., =10000. ? (the boundary is getting more complex)
 - accuracy: 0.6160, 0.6160, 0.8213, 0.8925 then if we use 'full data', it reaches **99%**
 - training time: 0.201 s
 - predicting time: 2.134 s
```
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=10.)

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" 

t0 = time()  
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test); accuracy 
```

## 00. main project
https://www.cs.cmu.edu/~enron/
Mine the email inboxes and financial data of Enron to identify **persons of interest** in one of the greatest corporate fraud cases in American history. We want to see if there are **any patterns within the emails of people** who were persons of interest in the fraud case and see if we can identify those people....Navigate through the data and extract some information !
> What is a **person of interest** ? 
 - indicted
 - settled w/o admitting guilt
 - testified in exchange for immunity

The Enron fraud is a big, messy and totally fascinating story about corporate malfeasance of nearly every imaginable type. The Enron email and financial datasets are also big, messy treasure troves of information, which become much more useful once you know your way around them a bit. We’ve combined the email and finance data into a single dataset. The aggregated Enron email + financial dataset is stored in a dictionary, where each key in the dictionary is a person’s name and the value is a dictionary containing all the features of that person. The email + finance (E+F) data dictionary is stored as a pickle file, which is a handy way to store and load python objects directly. 






































