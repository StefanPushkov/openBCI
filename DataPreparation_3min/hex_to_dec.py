import csv
import pandas as pd

def hex_to_dec(filename:str, out_dir: str):
    num = filename[-6:-4]
    with open(filename, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split() for line in stripped if line)
        final_rows = []
        for i in lines:
            new_line = []
            for item in i:
                new_line.append(int(item, 16))
            final_rows.append(new_line)
        out = '{1}/decimal{0}.csv'.format(num, out_dir)
        with open(out, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow([1, 2, 3, 4, 5, 6, 7, 8, 0])
            writer.writerows(final_rows)
    return out #'../prepared_data/decimal{0}.csv'.format(num)