# -*- coding: utf-8 -*-
'''

In this assignment:
·        You will learn how to use Scikit-learn
·        You will compare the performance of the Decision Tree model to the Random Forest (ensemble) model on a dataset
·        Learn how to build a classifier
·        Learn how to get the Accuracy, Precision, Recall, and F1 score to evaluate your model
·        Learn how to generate a confusion matrix

Dataset

In this assignment, you will apply both Decision Trees and Random Forest to a well-known classification dataset, the Iris dataset, to get a better feel for how Decision Trees 
and Random Forests work in practice. You can find the original dataset in the UCI ML repository. The Iris dataset has three classes, each with 50 instances, 
but we want to focus on the basic binary classification scenario, so we have removed one class of irises, Setosa, leaving just Versicolor and Virginica. 
The full dataset includes four attributes, sepal length and width, and petal length and width but in this assignment we will use only Sepal Length and petal width as attributes.
 
 
Instructions
1.  Split your dataset into training and test set using the sklearn.model_selection.train_test_split. 
    Let the training set have 70% of the data and the test set have 30% of the data.
2.  Train sklearn.tree.DecisionTreeClassifier on the training data using the fit method. 
    We are using the default parameter settings and letting DecisionTreeClassifier initialize its own weights (coefficients) to random values close to zero.
3.  Generate a classification report using sklearn.metrics.classification_report
4.  Generate a Confusion Matrix for your classification results using sklearn.metrics.confusion_matrix
5.  Now repeat Steps 1 – 4 (do not modify your code, rewrite it) but this time, rather than use sklearn.tree.DecisionTreeClassifier use sklearn.ensemble.RandomForestClassifier
6.  As a multiline comment in Python, answer the following questions
7.  How does the accuracy of the Decision Tree compare with that of the Random Forest?
8.  Why do you think one performed better than the other?

'''


import subprocess
import sys

def install(package):
  try:
    __import__(package)
  except ImportError:      
      subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("pandas")
install("sklearn")

#import libraries
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression,LinearRegression,Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.model_selection import train_test_split
pd.options.display.max_rows=12

#load training dataframes
sl_c = pd.read_csv("iris/train/iris_versicolor_vs_virginica.sl.c.train.csv")
sl_pl = pd.read_csv("iris/train/iris_versicolor_vs_virginica.sl.pl.train.csv")
sl_pw_c = pd.read_csv("iris/train/iris_versicolor_vs_virginica.sl_pw.c.train.csv")
sl_pw_pl = pd.read_csv("iris/train/iris_versicolor_vs_virginica.sl_pw.pl.train.csv")
#load dev set
sl_c_d = pd.read_csv("iris/tune/iris_versicolor_vs_virginica.sl.c.tune.csv")
sl_pl_d = pd.read_csv("iris/tune/iris_versicolor_vs_virginica.sl.pl.tune.csv")
sl_pw_c_d = pd.read_csv("iris/tune/iris_versicolor_vs_virginica.sl_pw.c.tune.csv")
sl_pw_pl_d = pd.read_csv("iris/tune/iris_versicolor_vs_virginica.sl_pw.pl.tune.csv")

#update class labels
sl_c.loc[sl_c["iris class"]=="Iris-virginica","iris class"]=1
sl_c.loc[sl_c["iris class"]=="Iris-versicolor","iris class"]=0
sl_c["iris class"]=sl_c["iris class"].astype('int')

sl_pw_c.loc[sl_pw_c["iris class"]=="Iris-virginica","iris class"]=1
sl_pw_c.loc[sl_pw_c["iris class"]=="Iris-versicolor","iris class"]=0
sl_pw_c["iris class"]=sl_pw_c["iris class"].astype('int')


sl_c_d.loc[sl_c_d["iris class"]=="Iris-virginica","iris class"]=1
sl_c_d.loc[sl_c_d["iris class"]=="Iris-versicolor","iris class"]=0
sl_c_d["iris class"]=sl_c_d["iris class"].astype('int')

sl_pw_c_d.loc[sl_pw_c_d["iris class"]=="Iris-virginica","iris class"]=1
sl_pw_c_d.loc[sl_pw_c_d["iris class"]=="Iris-versicolor","iris class"]=0
sl_pw_c_d["iris class"]=sl_pw_c_d["iris class"].astype('int')

print(sl_pw_c)

labels_ids = {'Iris-versicolor': 0, 'Iris-virginica': 1}

X = sl_pw_c[["sepal length", "petal width"]]
Y = sl_pw_c[["iris class"]]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

classifier = DecisionTreeClassifier()
y_predict = classifier.fit(x_train, y_train).predict(x_test)


print(classification_report(y_test, y_predict, target_names=labels_ids))

print(confusion_matrix(y_test, y_predict))

classifier = RandomForestClassifier()
y_predict = classifier.fit(x_train, y_train).predict(x_test)

print(classification_report(y_test, y_predict, target_names=labels_ids))

print(confusion_matrix(y_test, y_predict))

"""
How does the accuracy of the Decision Tree compare with that of the Random Forest?
The Random Forest method was more accurate, Decision Tree averaged 94% precision while
the Random Forest was at 100%

Why do you think one performed better than the other?
Since we are classifying between two binary options and with fewer differentiating aspects
the random forest performed better since there were not many decisions for the decision tree
to base its classification on

"""
