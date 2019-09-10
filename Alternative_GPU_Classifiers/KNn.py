from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import pandas as pd
import config as cf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os


data = pd.read_csv(cf.base_dir+cf.prepared_data_15min)
X = data.drop(['0'], axis=1)#[['1', '4', '7', '8']] # 1, 4, 7, 8
print(X.shape[0])
y = data[['0']].values.ravel()
# Feature Scaling
StdScaler = StandardScaler()
X_scaled = StdScaler.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.001, random_state=42)

clf = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan')
'''
grid_params = {
    'n_neighbors': [3, 5, 11], 
    'weights': ['uniform', 'distance'], 
    'metric': ['euclidean', 'manhattan']
}

gs = GridSearchCV(clf, grid_params, verbose=10, cv=2, n_jobs=-1)
gs.fit(X_Train, Y_Train)

print(gs.best_score_, gs.best_params_)
'''

clf.fit(X_Train, Y_Train)

# Save Model 
alt_models_dir = "/alt_models"
if not os.path.exists(alt_models_dir):
    os.makedirs(alt_models_dir)
dump(clf, cf.base_dir+alt_models_dir+'/KNN_model.joblib')

# Prediction 
pred = clf.predict(x_test)
#pred_P = neigh.predict_proba(x_test)
#print(pred_P)
ac = accuracy_score(y_test, pred)*100

print(ac)
