from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import pandas as pd
import config as cf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os


data = pd.read_csv(cf.base_dir+cf.prepared_data_3min)
X = data.drop(['0'], axis=1) #[['1', '4', '7', '8', '2']]
y = data[['0']].values.ravel()

# Feature Scaling
StdScaler = StandardScaler()
X_scaled = StdScaler.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.9, random_state=42)

model = load(cf.base_dir+'/alt_models/KNN_model.joblib')

pred = model.predict(x_test)

accu_percent = accuracy_score(y_test, pred) * 100
print("Accuracy obtained over the whole test set is %0.6f %% ." % (accu_percent))