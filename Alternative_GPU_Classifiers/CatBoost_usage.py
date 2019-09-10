import catboost as cb
import catboost.datasets as cbd
import catboost.utils as cbu
import numpy as np
import pandas as pd
import tensorflow as tf  # Just for checking if GPU is available :)
import config as cf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Checking if GPU is available
GPU_AVAILABLE = tf.test.is_gpu_available()
print("GPU available:", GPU_AVAILABLE)


# Get csv data
data = pd.read_csv(cf.base_dir+cf.prepared_data_3min)

X = data.drop(['0'], axis=1)
y = data[['0']].values.ravel()

# p_data = y.value_counts()
#print(p_data)

# Feature Scaling
StdScaler = StandardScaler()
X_scaled = StdScaler.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.99, random_state=0)

estimator = cb.CatBoostClassifier()
estimator.load_model(cf.base_dir+"/models/CatBoost.mlmodel")
pred = estimator.predict(x_test)

ac = accuracy_score(y_test, pred)

print(ac)
