import pandas as pd
import statistics
from openBCI import config as cf

import pandas as pd
import scipy.stats

def ent(data):
    data = pd.read_csv(data)

    # Read features and labels
    X = data.drop(['0'], axis=1)
    print(X.count())
    y = data['0']#.values.ravel()
    """Calculates entropy of the passed `pd.Series`
    """

    print(type(y))
    p_data = y.value_counts()
    print(p_data)# counts occurrence of each value
    entropy = scipy.stats.entropy(p_data)  # get entropy from counts
    return entropy

a = ent(cf.prepared_data_15min)
print(a)
'''
def count_variance(datafile: str):
    # Read data
    data = pd.read_csv(datafile)

    # Read features and labels
    X = data.drop(['0'], axis=1)
    y = data[['0']]#.values.ravel()

    # Get variance of the Dataframe:
    channel_variance = list(X.var())

    chan_var_list = list(enumerate(channel_variance, 1))
    sorted_list = sorted(chan_var_list, key=lambda item: item[1], reverse=True)
    print(sorted_list)
    selected_channels = []
    for i in range(8):
        selected_channels.append(str(sorted_list[i][0]))
    return selected_channels

a = count_variance(cf.prepared_data_3min)
print(a)

'''