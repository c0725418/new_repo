# https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python
# https://machinelearningmastery.com/feature-selection-machine-learning-python/

##Visualizations
#https://www.kaggle.com/alihannka/logistic-regression-with-gender-voice-prediction

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plot
import seaborn as sns 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn import metrics

"""
1.
First, we have to import the the voice dataset and check whether is loaded correctly
"""
#1.1 Reading and printing data in CSV file.
print(os.getcwd())
os.chdir("D:\Lambton College\Term 3\Introduction to Artificial Intelligence\Labs\Lab 01")
dataset_voice = pd.read_csv('voice.csv')
print (dataset_voice)
print (dataset_voice.info)
print(dataset_voice.head())
print (dataset_voice.shape)
 
genre_count = dataset_voice['label'].value_counts()
print(genre_count)


"""
2. 
all the features are going to be saved in the X array, whereas the values for 
target variable is going to be stored in y array
"""
#2.2 slicing the data
X = dataset_voice.iloc[:,:-1].values # all the rows, columns 2 and 3, starting from 0
y = dataset_voice.iloc[:,20].values
print(X)
print (y)


"""
3.
Since the target variable is a String, we have to convert it to data that can be 
read by our model. Since there are only 2 different string variables, we only
applied LabelEncoder. The values will be only 1 for male and 0 for female.
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder = LabelEncoder()
print (y)
y = LabelEncoder.fit_transform(y)
print (y)
#onehotencoder = OneHotEncoder(categorical_features=[2])
#y = onehotencoder.fit_transform(y).toarray()
#print (y)



"""
4. Split the data set 
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

"""
5. Standardize the train and data set. 
"Data standardization is the process of rescaling one or more attributes so 
that they have a mean value of 0 and a standard deviation of 1."
Source: 
https://machinelearningmastery.com/normalize-standardize-machine-learning-data-weka/
"""
from sklearn.preprocessing import StandardScaler
plot.hist(X_train)
plot.hist(X_test)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
plot.hist(X_train)
plot.hist(X_test)

"""
5. Logistic regression
"""
from sklearn.linear_model import LogisticRegression

#classifier = LogisticRegression()



classifier = LogisticRegression(random_state=0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print (pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print(metrics.accuracy_score(y_test, y_pred))

print(dataset_voice.corr())
corr = dataset_voice.corr() 
plot.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True, square = True, cmap = 'coolwarm')
plot.show()

"""
Feature selection and test again
In this case we used RFE class to remove columns and build a model using the 
features that remain
"""
from sklearn.feature_selection import RFE
classifier_2 = LogisticRegression()
rfe = RFE(classifier_2, 5)
fit = rfe.fit(X_train, y_train)
print("Num Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
#print (type(fit.support_))
indexes  = np.where(fit.support_)[0]
print (indexes)
#print(dataset_voice.iloc[:,indexes].values)
 

X = dataset_voice.iloc[:,indexes].values # all the rows, columns 2 and 3, starting from 0
print(X)
print (X.shape)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#print(X_train)
X_test = sc.fit_transform(X_test)
#print(X_test)
#print(y_train)
#print(y_test)


classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)

print(cm)
print (pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

print(metrics.accuracy_score(y_test, y_pred))



