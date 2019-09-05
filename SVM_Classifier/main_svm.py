import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from openBCI  import config as cf
#import matplotlib.pyplot as graph




# Function to compute the classification using SVM
def compute_SVC(X_train, y_train):
    c = svm.SVC(kernel='linear', gamma=0.005, C=0.05, verbose=10, probability=True)
    c.fit(X_train, y_train)
    return c


# Function to calculate the accuracy
def compute_accuracy(X_test, y_test, c):
    pred = c.predict(X_test)
    pred_accu = accuracy_score(y_test, pred)
    return pred_accu


# Function to compute the confusion matrix
def compute_confusion_matrix(test_f, test_l, c):
    pred = c.predict(test_f)
    x = confusion_matrix(test_l, pred)
    return x


data = pd.read_csv(cf.prepared_data_15min)
print(data.head())
# data = data.loc[:75055]

X = data.drop(['0'], axis=1)
y = data[['0']].values.ravel()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3) # when using .csv file


print('SVM is fitting......')
model_svc = compute_SVC(X_train, y_train)

print('Accuracy metric is testing')
accu_percent = compute_accuracy(X_test, y_test, model_svc) * 100
print("Accuracy obtained over the whole training set is %0.6f %% ." % (accu_percent))
#conf_mat = compute_confusion_matrix(features_train, labels_train, model_svc)
#print('Confusion matrix: ', conf_mat)
dump(model_svc, '../models/SVM_EEG.joblib')
