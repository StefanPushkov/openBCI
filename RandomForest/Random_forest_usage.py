from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import config as cf
from numpy.fft import fft
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import time
from FFT.wavelet_transform import feature_extraction
import pywt
import seaborn as sns; sns.set()
from datetime import datetime
#from Channel_selection import variance



def prediction(data: str):
    model = load(cf.base_dir+'/models/RandomForest_model_{0}.joblib'.format('real'))
    dt = pd.read_csv(data)
    # Get the channel numbers with the highest variance
    # data = data.loc[]
    X = dt.drop(['0'], axis=1).loc[:2000] #[['1', '4']]

    y = dt[['0']].loc[:2000]#.values.ravel()
    # X = np.c_[X]


    # Feature Scaling
    # StdScaler = StandardScaler()
    # X_scaled = StdScaler.fit_transform(X)


    Y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = Y_bin.shape[1]
    print(n_classes)
    X_Train, x_test, Y_Train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)

    print('Prediction started')
    start_time = time.time()  # Time counter
    print(" Started at ", datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'))

    X_Train_Wt, x_test_wt = feature_extraction(X_Train, x_test)
    print(X_Train_Wt.shape, x_test_wt.shape)
    print(X_Train_Wt[:1, :])
    """
    pred = model.predict_proba(x_test_wt)
    
    # Plot the micro-averaged Precision-Recall curve
    # For each class
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
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))
    plt.savefig(cf.base_dir+'/Plots/Avg_Prec_score.png')
    
    # Evaluating accuracy using Accuracy and Balanced Accuracy Score metrics
    p = model.predict(x_test_wt)
    print('Accuracy metrics are evaluated')

    # Accuracy
    accu_percent = accuracy_score(y_test, p) * 100
    print("Accuracy obtained over the whole test set is %0.6f %% ." % (accu_percent))

    # Balanced Accuracy Score
    # blnc = balanced_accuracy_score(y_test, p) * 100
    # print("balanced_accuracy_score: %0.6f %% ." % (blnc))

    # Confusion matrix
    classes = ['rest(0)', 'left', 'right']
    cm = confusion_matrix(y_test, p, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, ax=ax, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=classes, yticklabels=classes, linewidths=0.2, annot_kws={"size": 16})
    plt.xlabel('Truth')
    plt.ylabel('Predicted')
    plt.savefig(cf.base_dir + '/Plots/Conf_Matrix_RandomForest.png')
    plt.show()
    print("Conf_matrix: ", cm)

    # PLOT FOR EACH CLASS
    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'darkorange', 'green', 'red', 'blue'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig(cf.base_dir+'/Plots/Prec_Rec_curve_multi.png')

    print("Plot and prediction completed: %s seconds " % (time.time() - start_time))
    """

prediction(cf.base_dir+cf.prepared_data_imagery_V)

