raw_data_3min = "/data/OBCI_AE_SucksAssFull_LED.TXT"
raw_data_15min = "/data/OBCI_29_SucksAssFull.TXT"
#raw_data_15min = "/data/OBCI_AD_SucksAssFull_BtnPressed.TXT"
prepared_data_3min = "/prepared_data_3min/decimal30.csv"
prepared_data_15min = "/prepared_data_15min/combined_15min.csv"
#prepared_data_15min = "/prepared_data_15min/decimal29.csv"
#prepared_data_15min = "/prepared_data_15min/decimal31.csv"

margin_for_markup_3min = 0
margin_for_markup_15min = 820
BASE_DIR = 'openBCI'

import os
def get_base_dir_by_name(name):
    path = os.getcwd()
    lastchar = path.find(name) + len(name)
    return os.getcwd()[0:lastchar]

base_dir = get_base_dir_by_name(BASE_DIR).replace("\\","/")
