import config as cf
def convert_to_txt(filename: str):
    import csv
    num = filename[-6:-4]
    csv_file = filename
    txt_file = cf.base_dir+'/converted_data/full_prepared{0}.txt'.format(num)
    with open(txt_file, "w") as my_output_file:
        with open(csv_file, "r") as my_input_file:
            [my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()
    print(txt_file)
    return txt_file # 'E:/exp/converted_data/full_prepared{0}.txt'.format(num)
