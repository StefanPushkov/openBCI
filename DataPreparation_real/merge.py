import csv
import config as cf
reader1 = csv.reader(open(cf.base_dir + "/prepared_data_real/decimal30.csv"))
reader2 = csv.reader(open(cf.base_dir + "/prepared_data_real/decimal32.csv"))
reader3 = csv.reader(open(cf.base_dir + "/prepared_data_real/decimal34.csv"))
reader4 = csv.reader(open(cf.base_dir + "/prepared_data_real/decimal36.csv"))
reader5 = csv.reader(open(cf.base_dir + "/prepared_data_real/decimal38.csv"))
reader6 = csv.reader(open(cf.base_dir + "/prepared_data_real/decimal40.csv"))
f = open(cf.base_dir+"/prepared_data_real/combined_real.csv", "w")

writer = csv.writer(f)

for row in reader1:
    writer.writerow(row)
for row in reader2:
    writer.writerow(row)
for row in reader3:
    writer.writerow(row)
for row in reader4:
    writer.writerow(row)
for row in reader5:
    writer.writerow(row)
for row in reader6:
    writer.writerow(row)
f.close()