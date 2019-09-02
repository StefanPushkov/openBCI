from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from openBCI import config as cf
from sklearn.model_selection import train_test_split
import time
from datetime import datetime


def cv_RanfomForest():
    data = pd.read_csv(cf.prepared_data_15min)
    X = data.drop(['0'], axis=1)
    y = data[['0']].values.ravel()

    # Feature Scaling
    StdScaler = StandardScaler()
    X_scaled = StdScaler.fit_transform(X)

    # Splitting the dataset into the Training set and Test set
    X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    clf = RandomForestClassifier(random_state=0)

    # import multiprocessing
    # cores = multiprocessing.cpu_count()-1
    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                                   n_iter=20, cv=3, verbose=10, random_state=42)

    # Fit the random search model
    start_time = time.time() # Time counter
    print("Started at ", datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'))
    rf_random.fit(X_Train, Y_Train)
    best_p = rf_random.best_params_
    best_r = rf_random.best_score_

    print("Fitting time: %s seconds " % (time.time() - start_time))

    def cv_result(name: str):
        import json
        with open("../CV_result/cv_randomForest.txt", "w") as f:
            f.write(
                'Parameters used for Randomized grid search on ' + name + '\'s dataset: \nn_iter: ' + str(
                    rf_random.n_iter) + "\ncv: " + str(rf_random.cv))
            f.write('\nBest Params: \n')
            f.write(json.dumps(best_p))
            f.write('\nBest Accuracy: \n')
            f.write(json.dumps(best_r))
            f.close()

    cv_result(name='Mark')


cv_RanfomForest()
print('Hyperparameters tuning for Random Forest Classifier is completed')