# study-ML-intro

<img src="https://user-images.githubusercontent.com/31917400/34910969-bf8b8bee-f8b7-11e7-9d23-8d54d974c128.jpg" />

<img src="https://user-images.githubusercontent.com/31917400/35225294-d059a978-ff7f-11e7-9779-b3b51cca7549.jpg" />


------------------------------------------------------
## 3. Support Vector Machine
**[Find a DecisionSurface!]**
> PREDICTION: 
<img src="https://user-images.githubusercontent.com/31917400/39476600-673208d2-4d54-11e8-8f73-18871433f89c.jpg" />

SVM is a set of supervised learning methods used for 
 - classification
 - regression  
 - **outliers detection**

Pros & Cons
 - > The advantages of support vector machines are:
   - Effective in cases where number of dimensions is greater than the number of samples.
   - Uses a subset of training points in the decision function called `support vectors`, so it is also memory efficient.
   - Versatile: different **Kernel functions** can be specified for the decision function(Common kernels are provided, but it is also possible to specify custom kernels). 
   - Using a **kernel trick**, Linear DecisionSurf -> NonLinear DecisionSurf    

 - > The disadvantages of support vector machines include:
   - If the number of features is much greater than the number of samples, avoid **over-fitting** in choosing Kernel functions and **regularization term** is crucial.
   - SVMs do not directly provide probability estimates, these are calculated using an expensive **five-fold cross-validation**.
<img src="https://user-images.githubusercontent.com/31917400/35055161-61987186-fba6-11e7-8c97-b66617e8161c.jpg" width="750" height="150" />

Margine is a maximum distance to each nearest point. The separating line should be most robust to classification errors. The margine aims to maximizes the robustness of the result....As Much Separation b/w two classifications as possible. 
> The perceptron algorithm is a trick in which we started with a random line, and iterated on a step in order to slowly walk the line towards the misclassified points, so we can classify them correctly. However, we can also see this algorithm as an algorithm which minimizes an error function. 
<img src="https://user-images.githubusercontent.com/31917400/40259702-298552a2-5aef-11e8-9820-21406a2e0386.jpg" />

 - Error (Margin Error + Classification Error)
<img src="https://user-images.githubusercontent.com/31917400/40268051-8b7dc1b4-5b5e-11e8-8604-bb5e4468e452.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/40268052-8f949516-5b5e-11e8-8efc-d44acfa0eee3.jpg" />
 



```
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import copy

import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()
```
In sklearn.svm, `SVC()`, `NuSVC()`, `LinearSVC()` accept slightly different sets of parameters and have different mathematical formulations, but take as input two arrays: 
 - an array **X** of size `[n_samples, n_features]`holding the training samples 
 - an array **y** of class labels (strings or integers), size `[n_samples]`
 - Library: sklearn.svm 
 - Example: 
```
from sklearn.svm import SVC
clf = SVC(kernel="linear")
X = features_train
y = labels_train
clf.fit(X, y)

pred = clf.predict(features_test)
```
Accuracy ?
```
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
```

### Non-Linear SVM
<img src="https://user-images.githubusercontent.com/31917400/39048647-2962d1ba-4496-11e8-82ee-b87365d27b07.jpg" />  

 - Introducing New Features 'Z' or 'transformed X or Y' causes 'hyperplane.' Z is non-negative because it's a distance from the origin. 
<img src="https://user-images.githubusercontent.com/31917400/35122461-b94d14f4-fc96-11e7-9e22-1e3a76e58e16.jpg" /> 

 - Kernel Trick: There are functions taking a low dimensional given 'input space' and the added 'feature space' then map it to a very high dimensional space - Kernel function (Linear, rbf, poly, sigmoid). It makes the separation then takes the solution and go back to the original space. It sets the dataset apart where the division line is non-linear.
<img src="https://user-images.githubusercontent.com/31917400/35122799-e8106e2a-fc97-11e7-8872-43e13edacfd9.jpg" width="500" height="100" />

 - parameters (Kernel / Gamma / C)
   - Gamma: This parameter defines **how far the influence of a single data pt reaches**, with low values meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as the inverse of the radius of influence of samples selected by the model as support vectors. High gamma just like me..only thinking of sth right in my face. 
   - C: The 'gamma' parameter actually has no effect on the 'linear' kernel for SVMs. The key parameter for 'linear kernel function' is "C". The C parameter **trades off misclassification of training examples against simplicity of the decision surface**. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly by giving the model freedom to select more samples as support vectors - wiggling around individual data pt...
   - When gamma is very small, the model is too constrained and cannot capture the complexity or “shape” of the data. The region of influence of any selected support vector would include the whole training set. The resulting model will behave similarly to a linear model with a set of hyperplanes that separate the centers of high density of any pair of two classes. If gamma is too large, the radius of the area of influence of the support vectors only includes the support vector itself and no amount of regularization with C will be able to prevent overfitting. 
<img src="https://user-images.githubusercontent.com/31917400/35127560-923ca17c-fcaa-11e7-81ca-e4db864ccc96.jpg" /> 

SVMs "doesn't work well with lots and lots of noise, so when the classes are very overlapping, you have to count independent evidence.

>Naive Bayes is great for 'text'. It’s faster and generally gives better performance than an SVM. Of course, there are plenty of other problems where an SVM might work better. Knowing which one to try when you’re tackling a problem for the first time is part of the art of ML. 

In SVM, tuning the parameters can be a lot of work, but just sit tight for now--toward the end of the class we will introduce you to GridCV, a great sklearn tool that can find an optimal parameter tune almost automatically.

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






































