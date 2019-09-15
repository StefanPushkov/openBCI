import catboost as cb
from catboost import cv, Pool
import pandas as pd
import tensorflow as tf  # Just for checking if GPU is available :)
import config as cf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.fft import fft, rfft


# Checking if GPU is available
GPU_AVAILABLE = tf.test.is_gpu_available()
print("GPU available:", GPU_AVAILABLE)


# Get csv data
data = pd.read_csv(cf.base_dir+cf.prepared_data_real_comb)

X = data.drop(['0'], axis=1)
y = data[['0']].values.ravel()

# p_data = y.value_counts()
# print(p_data)

# Feature Scaling
StdScaler = StandardScaler()
X_scaled = StdScaler.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=0)


#estimator = cb.CatBoostClassifier(depth=10, iterations=600, verbose=10, task_type='GPU', learning_rate=0.08, allow_writing_files=False, loss_function='MultiClass')
estimator = cb.CatBoostClassifier(l2_leaf_reg=0.1, depth=8, iterations=1550, verbose=10, task_type='GPU', learning_rate=0.0775, allow_writing_files=False)

estimator.fit(X_Train, Y_Train)

'''
# Cross Validation
params = {'l2_leaf_reg': 0.1,
        'depth': 8,
              'iterations': 1550,
                         'verbose': 10,
                                 'task_type': 'GPU',
                                           'learning_rate': 0.0775}

cv_dataset = Pool(data=X_Train,
                  label=Y_Train)
scores = cv(cv_dataset,
            params,
            fold_count=2)

print('CV Result: ', scores)
'''

pred = estimator.predict(x_test)

# Saving model
print("Saving model...")
estimator.save_model(cf.base_dir+'/models/CatBoost.mlmodel')
ac = accuracy_score(y_test, pred)

print(ac)

