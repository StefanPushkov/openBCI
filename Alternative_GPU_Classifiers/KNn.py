from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import pandas as pd
from openBCI import config as cf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


data = pd.read_csv(cf.prepared_data_15min)

X = data.drop(['0'], axis=1)
y = data[['0']].values.ravel()

# Feature Scaling
StdScaler = StandardScaler()
X_scaled = StdScaler.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=0)


neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
neigh.fit(X_Train, Y_Train)

pred = neigh.predict(x_test)
pred_P = neigh.predict_proba(x_test)
print(pred_P)
ac = accuracy_score(y_test, pred) *100

print(ac)
'''
data_new = pd.read_csv(cf.prepared_data_3min)
X = data_new.drop(['0'], axis=1)
y = data_new[['0']].values.ravel()

# Feature Scaling
StdScaler = StandardScaler()
X_scaled = StdScaler.fit_transform(X)


pred_new = neigh.predict(X)


ac2 = accuracy_score(y, pred_new) *100

print('Test: ', ac2)
'''