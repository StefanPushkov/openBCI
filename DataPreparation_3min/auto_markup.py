import pandas as pd
import csv
import config as cf

def auto_markup(filename: str):
    data = pd.read_csv(filename, dtype='unicode')
    data.drop(data.tail(1).index, inplace=True) # Drop last line with NaN
    data_length = data.shape[0]
    data = data.loc[2:]

    index_list = [i for i in range(data_length - 2)]

    data.index = index_list
    data['class'] = 0
    num = filename[-9:-7]
    def set_class(start=0, stop=0, classname=0):
        data.loc[start:stop, 'class'] = classname

    left = data['left']
    right = data['right']

    def set_left_hand():
        left_down = []
        left_up = []
        for i in left.index:
            if left[i] == '1111':
                #data[i, 'class'] = 1
                left_down.append(i)

        for i in left.index:
            if left[i] == '2222':
                #data[i, 'class'] = 1
                left_up.append(i)

        left_interval = list(zip(left_down, left_up))
        for i in left_interval:
            set_class(start=i[0]-cf.margin_for_markup_3min, stop=i[1], classname=1)


    def set_right_hand():
        right_down = []
        right_up = []
        for i in right.index:
            if right[i] == '3333':
                # data[i, 'class'] = 1
                right_down.append(i)

        for i in right.index:
            if right[i] == '4444':
                # data[i, 'class'] = 1
                right_up.append(i)

        left_interval = list(zip(right_down, right_up))
        for i in left_interval:
            set_class(start=i[0]-cf.margin_for_markup_3min, stop=i[1], classname=2)

    set_left_hand()
    set_right_hand()
    data = data.drop(['N', 'right', 'left', 'time'], axis=1)
    out = "../converted_data/markup_class{0}.csv".format(num)
    data.to_csv(out, index=False, header=False)
    return out #"../converted_data/markup_class{0}.csv".format(num)