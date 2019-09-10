import csv
import config as cf
reader = csv.reader(open(cf.base_dir + cf.prepared_data_15min))
reader1 = csv.reader(open(cf.base_dir + "/prepared_data_15min/decimal31.csv"))
f = open(cf.base_dir+"/prepared_data_15min/combined_15min.csv", "w")
writer = csv.writer(f)

for row in reader:
    writer.writerow(row)
for row in reader1:
    writer.writerow(row)
f.close()