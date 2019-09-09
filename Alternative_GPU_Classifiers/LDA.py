### HAVEN'T GRID SEARCH

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import config as cf

# data_processing(cf.raw_data, 29)
# Get csv data
data = pd.read_csv(cf.base_dir+cf.prepared_data_15min)

X = data.drop(['0'], axis=1)
y = data['0']#.values.ravel()

p_data = y.value_counts()
print(p_data)
'''
# Feature Scaling
StdScaler = StandardScaler()
X_scaled = StdScaler.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# Init the Gaussian Classifier
clf = LinearDiscriminantAnalysis()

# Train the model
clf.fit(X_Train, Y_Train)

# Predict Output
pred = clf.predict(x_test)


# Plot Confusion Matrix
ac = accuracy_score(y_test, pred)
print(ac)
mat = confusion_matrix(pred, y_test)
print(mat)
names = (['rest(0)', 'left', 'right'])
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()
'''