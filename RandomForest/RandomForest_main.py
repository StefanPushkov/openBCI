import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import config as cf
from joblib import dump
from numpy.fft import fft
import numpy as np
#from Channel_selection.variance import count_variance

# from DataPreparation.main_preparation import data_processing

def RandomForest_fitting():
    # data_processing(cf.raw_data, 29)
    # Get csv data
    data = pd.read_csv(cf.base_dir+cf.prepared_data_real_comb)
    X = data.drop(['0'], axis=1) #[['1', '4']]

    print(X.shape)
    '''
    # Fourier Transform
    len = X.shape[0]
    print(len)
    ff = fft(X)
    ff = np.abs(ff[:len])
    X = np.transpose(np.array(ff), axes=[0, 1])
    
    # Get the channel numbers with the highest variance
    # channels = count_variance(cf.prepared_data_15min)

    # Get features from chosen channels
    # X = data[channels]
    y = data[['0']].values.ravel()

    # Feature Scaling
    StdScaler = StandardScaler()
    X_scaled = StdScaler.fit_transform(X)

    # Splitting the dataset into the Training set and Test set
    X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)



    # Fitting the classifier into the Training set
    clf = RandomForestClassifier(n_estimators=1000, min_samples_split=10, min_samples_leaf=1,
                                 max_features='sqrt', max_depth=70, bootstrap=False, random_state=0,
                                 verbose=10, n_jobs=-1)

    print('RandomForest fitting...')
    clf.fit(X_Train, Y_Train)

    # Predicting the test set results
    pred = clf.predict(x_test)

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