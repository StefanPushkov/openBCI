import xgboost as xgb
import numpy as np
import pandas as pd
from openBCI import config as cf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Get csv data
data = pd.read_csv(cf.prepared_data_15min)

X = data.drop(['0'], axis=1)
y = data[['0']].values.ravel()

# p_data = y.value_counts()
#print(p_data)

# Feature Scaling
StdScaler = StandardScaler()
X_scaled = StdScaler.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=0)

dtrain = xgb.DMatrix(X_Train, label=Y_Train)
dtest = xgb.DMatrix(x_test, label=y_test)

param = {'objective': 'multi:softmax', # Specify multiclass classification
         'num_class': 3, # Number of possible output classes
         'tree_method': 'gpu_hist' # Use GPU accelerated algorithm
         }
gpu_res = {} # Store accuracy result
tmp = time.time()
# Train model
xgb.train(param, dtrain, 300, evals=[(dtest, 'test')], evals_result=gpu_res)
print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))