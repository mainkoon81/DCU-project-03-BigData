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
np.random.seed(42)
```



















## 2. Naive Bayes
**To find a DecisionSurface !**    
 - Library: sklearn.naive_bayes (Gaussian Naive Bayes)
 
 - Example
   - Compute the accuracy of your Naive Bayes classifier. Accuracy is defined as the number of test points that are classified correctly divided by the total number of test points.
```
def NBAccuracy(features_train, labels_train, features_test, labels_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()    ### create classifier
    clf.fit(features_train, labels_train)    ### fit the classifier on the training features and labels
    pred = clf.predict(features_test)    ### use the trained classifier to predict labels for the test features

    ### calculate and return the accuracy on the test data. 
    accuracy = clf.score(features_test, labels_test)
    return(accuracy)
    
    ### or we can import 'sklearn accuracy'
    from sklearn.metrics import accuracy_score
    print(accuracy_score(pred, labels_test))
```
It throws an accuracy of 88.4% which means 88.4% of the points are being correctly labelled by our classifier-'clf' when we use our test-set ! 

>__Bayes Rule:__ 
<img src="https://user-images.githubusercontent.com/31917400/34920230-5115b6b6-f967-11e7-9493-5f6662f1ce70.JPG" />

Semantically, what Bayes rule does is it incorporates some evidence from the test into our **prior** to arrive at a **posterior**.
 - Prior: Probability before running a test.
 - test evidence
 - Posterior: 
<img src="https://user-images.githubusercontent.com/31917400/34921660-73b52056-f97d-11e7-87a8-d2aae8257248.jpg" />
































