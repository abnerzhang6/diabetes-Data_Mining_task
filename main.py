import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from Tools import *

print("Analysis begin...")
print("Data reading...")
diabetes = pd.read_csv("diabetes_prediction_dataset.csv")
# diabetes = diabetes.head(100)
print("Data cleaning...")
X, y = data_clean(diabetes)

print("Data cleaning finished")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# --------------------------------------------- #
print("Logistic Regression calculating...")
LogiReg = LogisticRegressionCV(
    cv=5, solver='liblinear').fit(X_train, y_train)
score = LogiReg.score(X_test, y_test)
print("Finished")
print(f"5-Fold Logistics Regression average accuracy: {score}")
print("------------------------------------------------")
# --------------------------------------------- #
print("Decision Tree calculating...")
DeciTree = DecisionTreeClassifier()
result = cross_val_score(DeciTree, X, y, cv=5)
score = np.mean(result)
print("Finished")
print(f"5-Fold Decision Tree average accuracy: {score}")
print("------------------------------------------------")
# --------------------------------------------- #
print("Random Forest calculating...")
RandFore = RandomForestClassifier()
result = cross_val_score(RandFore, X, y, cv=5)
score = np.mean(result)
print("Finished")
print(f"5-Fold Random Forest average accuracy: {score}")
print("------------------------------------------------")
# --------------------------------------------- #
print("KNN calculating...")
KNN = KNeighborsClassifier()
result = cross_val_score(KNN, X, y, cv=5)
score = np.mean(result)
print("Finished")
print(f"5-Fold KNN average accuracy: {score}")
print("------------------------------------------------")
# --------------------------------------------- #
print("Naive Bayes calculating...")
NaiveBayes = GaussianNB()
result = cross_val_score(NaiveBayes, X, y, cv=5)
score = np.mean(result)
print("Finished")
print(f"5-Fold Naive Bayes average accuracy: {score}")
print("------------------------------------------------")
# --------------------------------------------- #
print("SVM calculating...")
SVM = svm.SVC()
result = cross_val_score(SVM, X, y, cv=5)
score = np.mean(result)
print("Finished")
print(f"5-Fold SVM average accuracy: {score}")
print("------------------------------------------------")
# --------------------------------------------- #
print("AdaBoost calculating...")
AdaBoost = AdaBoostClassifier()
result = cross_val_score(AdaBoost, X, y, cv=5)
score = np.mean(result)
print("Finished")
print(f"5-Fold AdaBoost average accuracy: {score}")
print("------------------------------------------------")
# --------------------------------------------- #
