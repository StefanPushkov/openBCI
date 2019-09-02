from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from openBCI import config as cf
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import time
from datetime import datetime
from openBCI.Channel_selection import variance


def prediction(data: str):
    model = load('../models/RandomForest_model_15min.joblib')
    dt = pd.read_csv(data)
    # Get the channel numbers with the highest variance
    # data = data.loc[]
    X = dt.drop(['0'], axis=1)
    # channels = variance.count_variance(data)
    # X = dt[channels]
    y = dt.iloc[:,8:]#.values.ravel()
    #X = np.c_[X]


    # Feature Scaling
    StdScaler = StandardScaler()
    X_scaled = StdScaler.fit_transform(X)


    Y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = Y_bin.shape[1]
    print(n_classes)
    # X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, Y, test_size=0.99, random_state=0)

    print('Prediction started')
    start_time = time.time()  # Time counter
    print(" Started at ", datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'))


    pred = model.predict_proba(X_scaled)

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
    plt.savefig('../Plots/Avg_Prec_score.png')
    # Evaluating accuracy using Accuracy and Balanced Accuracy Score metrics
    p = model.predict(X_scaled)
    print('Accuracy metrics are evaluated')

    # Accuracy
    accu_percent = accuracy_score(y, p) * 100
    print("Accuracy obtained over the whole test set is %0.6f %% ." % (accu_percent))

    # Balanced Accuracy Score
    blnc = balanced_accuracy_score(y, p) * 100
    print("balanced_accuracy_score: %0.6f %% ." % (blnc))

    # Confusion matrix
    cm = confusion_matrix(y, p)
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
    plt.savefig('../Plots/Prec_Rec_curve_multi.png')

    print("Plot and prediction completed: %s seconds " % (time.time() - start_time))


prediction(cf.prepared_data_3min)

