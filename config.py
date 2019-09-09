raw_data_3min = "/data/OBCI_2B.TXT"
raw_data_15min = "/data/OBCI_29_SucksAssFull.TXT"
raw_data_new = "/data/OBCI_75.TXT"
prepared_data_3min = "/prepared_data_3min/decimal2B.csv"
prepared_data_15min = "/prepared_data_15min/decimal29.csv"
prepared_data_new = "/prepared_data_new/decimal75.csv"
margin_for_markup_3min = 0
margin_for_markup_15min = 620

import os
def get_base_dir_by_name(name):
    path = os.getcwd()
    lastchar = path.find(name) + len(name)
    return os.getcwd()[0:lastchar]

base_dir = get_base_dir_by_name('openBCI').replace("\\","/")
print(base_dir)