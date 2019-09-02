from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import config as cf
from sklearn.model_selection import train_test_split

data = pd.read_csv(cf.prepared_data_15min)
X = data.drop(['0'], axis=1)
y = data[['0']].values.ravel()

# Splitting the dataset into the Training set and Test set
X_Train, x_test, Y_Train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
StdScaler = StandardScaler()
X_Train = StdScaler.fit_transform(X_Train)

# Kernel types
kernel = ['linear', 'rbf']
# C parameter values
C = [0.05, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 5, 10]
# Gamma parameter values
gamma = [1/4000000, 1/2000000, 1/400000, 1/200000, 1/40000, 1/20000, 1/4000, 1/2000, 1/400, 1/200]


random_grid = {'C': C,
               'kernel': kernel,
               'gamma': gamma}

clf = svm.SVC(random_state=0)

rf_random = RandomizedSearchCV(estimator = clf, param_distributions=random_grid,
                               n_iter=4, cv=2, verbose=10, random_state=0, n_jobs=-1)

# Fit the random search model
rf_random.fit(X_Train, Y_Train)

best_p = rf_random.best_params_
best_r = rf_random.best_score_
print(best_p, best_r)

import json
with open("../CV_result/cv_SVM.txt", "w") as f:
    f.write('Best Params: \n')
    f.write(json.dumps(best_p))
    f.write('\nBest Accuracy: \n')
    f.write(json.dumps(best_r))
    f.close()

print('Hyperparameters tuning for SVM is completed')