import csv
import config as cf

def text_to_csv(filename: str, num: int):
    with open(filename, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        out = cf.base_dir+'/converted_data/{0}CSV.csv'.format(num)
        with open(out, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('N', '1', '2', '3', '4', '5', '6', '7', '8', "left", 'right', 'time'))
            writer.writerows(lines)
    print(out)
    return out #'E:/exp/converted_data/{0}CSV.csv'.format(num)

