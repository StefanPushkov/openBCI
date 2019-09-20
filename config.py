# Nums for fully prepared data filenames
# num_real_M = 28         # Ральные Марк
# num_imagery_M = 29      # Воображаемые Марк

num_real_V = 40     # Реальные Володя
num_imagery_V = 31      # Воображаемые Володя

nums_real_V_list = [30, 32, 34, 36, 38]
nums_imagery_V_list = [31]

#raw_data_imagery_M = "/data/OBCI_2B.TXT"                                 # Марк
#raw_data_real_M = "/data/OBCI_29_SucksAssFull.TXT"                   # Марк
raw_data_real_V = "/data/real/OBCI_BD_new_placement_real.TXT"       # Володя
raw_data_imagery_V = "/data/imagery/OBCI_AE_SucksAssFull_LED.TXT"               # Володя
prepared_data_real_V = "/prepared_data_real/decimal{0}.csv".format(num_real_V)         # Володя
prepared_data_imagery_V = "/prepared_data_imagery/decimal{0}.csv".format(num_imagery_V)            # Володя
prepared_data_real_comb = "/prepared_data_real/combined_real.csv"
#prepared_data_real_M = "/prepared_data_real/decimal{0}.csv".format(num_real_M)          # Марк

margin_for_markup_imagery = 0
margin_for_markup_real = 500
BASE_DIR = 'openBCI'

import os
def get_base_dir_by_name(name):
    path = os.getcwd()
    lastchar = path.find(name) + len(name)
    return os.getcwd()[0:lastchar]

base_dir = get_base_dir_by_name(BASE_DIR).replace("\\","/")


