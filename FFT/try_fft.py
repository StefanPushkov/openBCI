import pandas as pd
import config as cf
import numpy as np
from numpy.fft import fft, fft2, fftfreq
import matplotlib.pyplot as plt
data = pd.read_csv(cf.base_dir+cf.prepared_data_real_comb)
data = data.loc[:50000]

X = data[['3']]#.drop(['0'], axis=1)
y = data[['0']].values.ravel()
print(X.shape)

FD = 250 # частота дескретизации
N = X.shape[0] # количесвто строк в записи (frames)

spectrum = fft(X)
# print(type(spectrum))
# freqs = fftfreq(N, 1./FD)
# print(type(freqs))

'''
for coef,freq in zip(spectrum,freqs):
    if coef:
        print('{c:>6} * exp(2 pi i t * {f})'.format(c=coef,f=int(freq)))


'''



print(spectrum.shape)


# Амплитудно-верменное измерение
plt.plot(np.arange(X.shape[0])/float(FD), X, 'b') # отрисовка сигнала синим
plt.xlabel(u'Время, c') # это всё запускалось в Python 2.7, поэтому юникодовские строки
plt.ylabel(u'Напряжение, мВ')
plt.title(u'Cигнал и тон 250 Гц')
plt.grid(True)
plt.show()



# Cпектр (Частотно-Амплитудное измерение)
plt.plot(fftfreq(N, 1./FD), np.abs(spectrum)/N)
# rfftfreq сделает всю работу по преобразованию номеров элементов массива в герцы
# нас интересует только спектр амплитуд, поэтому используем abs из numpy (действует на массивы поэлементно)
# делим на число элементов, чтобы амплитуды были в милливольтах, а не в суммах Фурье. Проверить просто — постоянные составляющие должны совпадать в сгенерированном сигнале и в спектре
plt.xlabel(u'Частота, Гц')
plt.ylabel(u'Напряжение, мВ')
plt.title(u'Спектр')
plt.grid(True)
plt.show()





#a = np.abs(ffq[:75664])
#a = pd.Series((a))
#print(a)
#print(a.shape)