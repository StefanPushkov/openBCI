import catboost as cb
import numpy as np
import pandas as pd
import config as cf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import time
from datetime import datetime

def CatBoost_CV():
    # Get csv data
    data = pd.read_csv(cf.base_dir+cf.prepared_data_real_comb)

    X = data.drop(['0'], axis=1)
    y = data[['0']].values.ravel()

    # Feature Scaling
    StdScaler = StandardScaler()
    X_scaled = StdScaler.fit_transform(X)

    # Splitting the dataset into the Training set and Test set
    X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=0)

    # Number of trees in random forest
    iterations = [int(x) for x in np.linspace(start=200, stop=2000, num=5)]
    # Number of features to consider at every split
    learning_rate = [x for x in np.linspace(start=0.01, stop=0.1, num=5)]
    # Maximum number of levels in tree
    depth = [int(x) for x in np.linspace(6, 10, num=4)]
    # Minimum number of samples required to split a node
    l2_leaf_reg = [x for x in np.linspace(start=0.01, stop=0.1, num=5)]
    # Minimum number of samples required at each leaf node
   # per_float_feature_quantization = ['0:border_count=1024', '1:border_count=1024']
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {'iterations': iterations,
                   'learning_rate': learning_rate,
                   'depth': depth,
                   'l2_leaf_reg': l2_leaf_reg}
                   #'per_float_feature_quantization': per_float_feature_quantization}

    clf = cb.CatBoostClassifier(random_state=0, border_count=255, task_type='GPU')

    # import multiprocessing
    # cores = multiprocessing.cpu_count()-1
    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                                   n_iter=5, cv=3, verbose=10, random_state=42, n_jobs=1)

    # Fit the random search model
    start_time = time.time()  # Time counter
    print("Started at ", datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'))
    rf_random.fit(X_Train, Y_Train)
    best_p = rf_random.best_params_
    best_r = rf_random.best_score_

    print(best_p, best_r)
CatBoost_CV()