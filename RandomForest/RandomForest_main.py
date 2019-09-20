import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import config as cf
import pywt
from joblib import dump
from numpy.fft import fft
from FFT.wavelet_transform import feature_extraction
import numpy as np
#from Channel_selection.variance import count_variance

# from DataPreparation.main_preparation import data_processing



def RandomForest_fitting():
    # Get csv data
    data = pd.read_csv(cf.base_dir+cf.prepared_data_real_comb)
    data = data.loc[:2000]
    X = data.drop(['0'], axis=1) #[['1', '4']]
    y = data[['0']].values.ravel()
    print("CSV X: ", type(X))

    # Feature Scaling
    StdScaler = StandardScaler()
    X_scaled = StdScaler.fit_transform(X)
    print("X_scaled: ", type(X_scaled))
    print('X_scaled: ', X_scaled[:2])
    # Splitting the dataset into the Training set and Test set
    X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)
    print('Splitted X: ', type(X_Train))
    print('Splitted X: ', X_Train[:2])
    X_Train_WT, x_test_wt = feature_extraction(X_Train, x_test)
    print('WT X: ', type(X_Train_WT))
    print('WT X: ', X_Train_WT[:2])

    '''
    # Fitting the classifier into the Training set
    clf = RandomForestClassifier(n_estimators=1000, min_samples_split=10, min_samples_leaf=1,
                                 max_features='sqrt', max_depth=70, bootstrap=False, random_state=0,
                                 verbose=10, n_jobs=-1)

    print('RandomForest fitting...')

    clf.fit(X_Train_WT, Y_Train)

    # Predicting the test set results

    pred = clf.predict(x_test_wt)

    # Model Saving
    dump(clf, cf.base_dir+'/models/RandomForest_model_{0}.joblib'.format('real'))

    # Testing accuracy
    print('Accuracy metrics are evaluated')

    # Accuracy
    accu_percent = accuracy_score(y_test, pred) * 100
    print("Accuracy obtained over the whole training set is %0.6f %% ." % (accu_percent))

    # Balanced Accuracy Score
    blnc = balanced_accuracy_score(y_test, pred) * 100
    print("balanced_accuracy_score: %0.6f %% ." % (blnc))
    '''

RandomForest_fitting()