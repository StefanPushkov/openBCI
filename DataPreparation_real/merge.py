import csv
import config as cf
reader = csv.reader(open(cf.base_dir + "/prepared_data_real/decimal30.csv"))
reader1 = csv.reader(open(cf.base_dir + "/prepared_data_real/decimal32.csv"))
reader2 = csv.reader(open(cf.base_dir + "/prepared_data_real/decimal34.csv"))
reader3 = csv.reader(open(cf.base_dir + "/prepared_data_real/decimal36.csv"))
reader4 = csv.reader(open(cf.base_dir + "/prepared_data_real/decimal38.csv"))
f = open(cf.base_dir+"/prepared_data_real/combined_real.csv", "w")

writer = csv.writer(f)

for row in reader:
    writer.writerow(row)
for row in reader1:
    writer.writerow(row)
for row in reader2:
    writer.writerow(row)
for row in reader3:
    writer.writerow(row)
for row in reader4:
    writer.writerow(row)
f.close()