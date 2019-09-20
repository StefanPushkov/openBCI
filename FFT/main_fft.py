import pandas as pd
import config as cf
from numpy.fft import fft, rfft
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

data = pd.read_csv(cf.base_dir+cf.prepared_data_real_V)
print(data.shape)

X = data.drop(['0'], axis=1)
y = data[['0']].values.ravel()

# Get real amplitudes of FFT (only in postive frequencies)
fft_vals = np.absolute(fft(X))
print(fft_vals.shape)

# Get frequencies for amplitudes in Hz
fft_freq = np.fft.rfftfreq(len(X), 1.0/fs)
print(fft_freq.shape)

# Define EEG bands
eeg_bands = {'Delta': (0, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 45)}

# Take the mean of the fft amplitude for each EEG band
eeg_band_fft = dict()
for band in eeg_bands:
    freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                       (fft_freq <= eeg_bands[band][1]))[0]
    eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
    print(eeg_band_fft[band])


# Plot the data (using pandas here cause it's easy)

df = pd.DataFrame(columns=['band', 'val'])
df['band'] = eeg_bands.keys()
df['val'] = [eeg_band_fft[band] for band in eeg_bands]
ax = df.plot.bar(x='band', y='val', legend=False)
ax.set_xlabel("EEG band")
ax.set_ylabel("Mean band Amplitude")
plt.show()


