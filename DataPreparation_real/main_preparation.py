from DataPreparation_real.txt_to_csv import text_to_csv
from DataPreparation_real.auto_markup import auto_markup
from DataPreparation_real.to_txt import convert_to_txt
from DataPreparation_real.hex_to_dec import hex_to_dec
import config as cf
import os


final_directory = cf.base_dir+"/prepared_data_real"
dir_for_converted_data = cf.base_dir+"/converted_data"
raw_data = cf.base_dir+cf.raw_data_real_V


def data_processing(raw_data: str, num_new=None, out_dir=final_directory):
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    if not os.path.exists(dir_for_converted_data):
        os.makedirs(dir_for_converted_data)
    print('Data processing started')
    # Function txt_to_csv returns 'C:/Storage/Mark/BCI/openBCI/converted_data/{0}CSV.csv'.format(num)
    csv = text_to_csv(raw_data, num_new)


    # Function auto_markup returns "C:/Storage/Mark/BCI/openBCI/converted_data/markup_class{0}.csv".format(num)
    marked_up = auto_markup(csv)


    # Function convert_to_txt returns 'C:/Storage/Mark/BCI/openBCI/converted_data/full_prepared{0}.txt'.format(num)
    txt = convert_to_txt(marked_up)


    # Function hex_to_dec returns 'C:/Storage/Mark/BCI/openBCI/prepared_data_3_min/decimal{0}.csv'.format(num)
    hex_to_dec(txt, out_dir=out_dir)
    print('Data processing ended')

# Merging to one
data_processing(raw_data=raw_data, num_new=cf.num_real_V, out_dir=final_directory)
