from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import config as cf
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import time
import seaborn as sns; sns.set()
from datetime import datetime

def a():
    model = load('../models/SVM_EEG.joblib')
    data = pd.read_csv(cf.base_dir+cf.prepared_data_imagery_V)
    X = data.drop(['0'], axis=1)
    y = data[['0']]  # .values.ravel()
    X = np.c_[X]
    print(y.shape)

    # Feature Scaling
    StdScaler = StandardScaler()
    X_scaled = StdScaler.fit_transform(X)

    Y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]
    print()
    # X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, Y, test_size=0.99, random_state=0)

    print('Prediction started')
    start_time = time.time()  # Time counter
    print(" Started at ", datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'))
    pred = model.predict_proba(X_scaled)
    p = model.predict(X_scaled)
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_bin[:, i],
                                                            pred[:, i])
        average_precision[i] = average_precision_score(Y_bin[:, i], pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_bin.ravel(),
                                                                    pred.ravel())
    average_precision["micro"] = average_precision_score(Y_bin, pred,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    plt.savefig('../Plots/Avg_Prec_scoreSVM.png')

    # Confusion matrix
    cm = confusion_matrix(y, p, labels=['rest(0)', 'left', 'right'])
    names = (['rest(0)', 'left', 'right'])
    sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=names, yticklabels=names)
    plt.xlabel('Truth')
    plt.ylabel('Predicted')
    plt.savefig('../Plots/Conf_Matrix_SVM.png')
    plt.show()
    print("Conf_matrix: ", cm)

    # Accuracy
    accu_percent = accuracy_score(y, p) * 100
    print("Accuracy obtained over the whole test set is %0.6f %% ." % (accu_percent))

    # Balanced Accuracy Score
    blnc = balanced_accuracy_score(y, p) * 100
    print("balanced_accuracy_score: %0.6f %% ." % (blnc))

a()