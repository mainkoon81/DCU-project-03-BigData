# study-ML-intro

<img src="https://user-images.githubusercontent.com/31917400/34910969-bf8b8bee-f8b7-11e7-9d23-8d54d974c128.jpg" />

## 1. Logistic Regression
<img src="https://user-images.githubusercontent.com/31917400/34471521-d497e2bc-ef43-11e7-8e70-5d232b659be0.jpg" />

**Typical Approach**
 - Fitting a logistic regression to a dataset where we would like to predict if a transaction is fraud or not.
<img src="https://user-images.githubusercontent.com/31917400/34495490-4fb25f96-efed-11e7-8fb0-5eadb50da2d0.jpg" width="160" height="50" />

As we can see, there are two columns that need to be changed to dummy variables. Use the 1 for weekday and True, and 0 otherwise.
```
df['weekday'] = pd.get_dummies(df['day'])['weekday']
df[['not_fraud','fraud']] = pd.get_dummies(df['fraud'])

df = df.drop('not_fraud', axis=1)
df.head(2)
```
<img src="https://user-images.githubusercontent.com/31917400/34495708-4c4fd206-efee-11e7-8a32-1f419d1aa80e.jpg" width="200" height="50" />

The proportion of fraudulent, weekday... transactions...?
```
print(df['fraud'].mean())
print(df['weekday'].mean())
print(df.groupby('fraud').mean()['duration'])
```
<img src="https://user-images.githubusercontent.com/31917400/34495836-e1ec77ba-efee-11e7-826c-fc707de638ce.jpg" width="120" height="50" />

Fit a logistic regression model to predict if a transaction is fraud using both day and duration. Don't forget an intercept! Instead of 'OLS', we use 'Logit'
```
df['intercept'] = 1

log_model = sm.Logit(df['fraud'], df[['intercept', 'weekday', 'duration']])
result = log_model.fit()
result.summary()
```
<img src="https://user-images.githubusercontent.com/31917400/34496037-d41f3d2e-efef-11e7-85b9-d88c9d2faa30.jpg" width="400" height="100" />

Coeff-interpret: we need to exponentiate our coefficients before interpreting them.
```
# np.exp(result.params)
np.exp(2.5465)
np.exp(-1.4637), 100/23.14
```
12.762357271496972, (0.23137858821179411, 4.32152117545376)

>On weekdays, the chance of fraud is 12.76 (e^2.5465) times more likely than on weekends...holding 'duration' constant. 

>For each min less spent on the transaction, the chance of fraud is 4.32 times more likely...holding the 'weekday' constant. 

*Note: When you find the ordinal variable with numbers...Need to convert to the categorical variable, then
```
df['columns'].astype(str).value_counts()
```

**Diagnostics**
```
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
```
 - __Confusion Matrix__
   - Recall: 'reality'(Out of all the items that are **truly positive**): TP / TP+FN
   - Precision 'argued'(Out of all the items **labeled as positive**): TP / TP+FP
<img src="https://user-images.githubusercontent.com/31917400/35222288-54f13ad0-ff75-11e7-819c-36eae7e370c1.jpg" />

 - Next, it is useful to split your data into training and testing data to assure your model can predict well not only on the data it was fit to, but also on data that the model has never seen before. Proving the model performs well on test data assures that you have a model that will do well in the future use cases.** 
```

```


## 2. Naive Bayes
**To find a DecisionSurface !**    
 - Library: sklearn.naive_bayes (Gaussian)
 - Example: Compute the accuracy of your Naive Bayes classifier. Accuracy is defined as the number of test points that are classified correctly divided by the total number of test points.
```
def NBAccuracy(features_train, labels_train, features_test, labels_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()    ### create classifier ###
    clf.fit(features_train, labels_train)    ### fit the classifier on the training features and labels ###
    pred = clf.predict(features_test)    ### use the trained classifier to predict labels for the test features ###

    ### calculate and return the accuracy on the test data. ### 
    accuracy = clf.score(features_test, labels_test)
    return(accuracy)
    
    ### or we can use 'sklearn accuracy' ###
    from sklearn.metrics import accuracy_score
    print(accuracy_score(pred, labels_test))
```
It throws an accuracy of 88.4% which means 88.4% of the points are being correctly labelled by our classifier-'clf' when we use our test-set ! 

>__Bayes Rule:__ 
<img src="https://user-images.githubusercontent.com/31917400/34920230-5115b6b6-f967-11e7-9493-5f6662f1ce70.JPG" width="400" height="500" />

*Semantically, what Bayes rule does is it **incorporates** some evidence from the test into our **prior** to arrive at a **posterior**.
 - Prior: Probability before running a test.
 - test evidence
 - Posterior: 
<img src="https://user-images.githubusercontent.com/31917400/34955056-b8ae9834-fa1a-11e7-8ceb-a593ed75361a.jpg" />

*Algorithm of Naive Bayes
<img src="https://user-images.githubusercontent.com/31917400/34954589-e3b3d3c0-fa18-11e7-8141-08e522668276.jpg" />

*Example_1. Text Forensic and Learning (ex. Whose email would it be ? Sent from Chris or Sara ?)
<img src="https://user-images.githubusercontent.com/31917400/34954221-88772364-fa17-11e7-9f46-fb3d91d94be3.jpg" />

## 3. Support Vector Machine
**To find a DecisionSurface !**    
SVM is a set of supervised learning methods used for classification, regression and **outliers detection**.
<img src="https://user-images.githubusercontent.com/31917400/35055161-61987186-fba6-11e7-8c97-b66617e8161c.jpg" width="750" height="150" />

Margine is a maximum distance to each nearest point. The separating line should be most robust to classification errors. The margine aims to maximizes the robustness of the result....As Much Separation b/w two classifications as possible. 

 - The advantages of support vector machines are:
   - Effective in cases where number of dimensions is greater than the number of samples.
   - Uses a subset of training points in the decision function called `support vectors`, so it is also memory efficient.
   - Versatile: different **Kernel functions** can be specified for the decision function(Common kernels are provided, but it is also possible to specify custom kernels).
   
 - The disadvantages of support vector machines include:
   - If the number of features is much greater than the number of samples, avoid **over-fitting** in choosing Kernel functions and **regularization term** is crucial.
   - SVMs do not directly provide probability estimates, these are calculated using an expensive **five-fold cross-validation**. 
```
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

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

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
```

### Non-Linear SVM
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
























