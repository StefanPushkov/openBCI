import statistics
import config as cf
import pandas as pd
import scipy.stats

def ent(data):
    data = pd.read_csv(data)

    # Read features and labels
    X = data.drop(['0'], axis=1)
    y = data['0']#.values.ravel()
    """Calculates entropy of the passed `pd.Series`
    """

    p_data = y.value_counts()
    print("Samples for each class: \n", p_data) # counts occurrence of each value
    entropy = scipy.stats.entropy(p_data)  # get entropy from counts
    return entropy


def count_variance(datafile: str, num=0):
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
    for i in range(4):
        selected_channels.append(str(sorted_list[i][0]))
    return selected_channels



def main(datafile):
    a = ent(datafile)
    print('Entropy value: ', a)
    b = count_variance(datafile)
    print(b)


main(cf.base_dir + cf.prepared_data_real_comb)