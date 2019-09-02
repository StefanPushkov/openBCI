from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
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
rf_random_r = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                                   n_iter=20, cv=3, verbose=10, random_state=42)

rf_random = {"n_estimators": 2000, "min_samples_split": 2, "min_samples_leaf": 2,
             "max_features": "auto", "max_depth": 50, "bootstrap": False}

def cv_result(name: str):
    import json
    with open("../CV_result/cv_randomForestTEST.txt", "w") as f:
        f.write(
            'Parameters used for Randomized grid search on ' + name + '\'s dataset: \nn_iter: ' + str(rf_random_r.n_iter) + "\ncv: " + str(rf_random_r.cv))
        f.write('\nBest Params: \n')
        #f.write(json.dumps(best_p))
        f.write('\nBest Accuracy: \n')
        #f.write(json.dumps(best_r))
        f.close()


cv_result(name='Mark')