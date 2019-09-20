import numpy as np
import pywt

def feature_extraction(X_Tr=None, x_t=None):
    def calculate_statistics(list_values):
        # n5 = np.nanpercentile(list_values, 5)
        # n25 = np.nanpercentile(list_values, 25)
        # n75 = np.nanpercentile(list_values, 75)
        # n95 = np.nanpercentile(list_values, 95)
        # median = np.nanpercentile(list_values, 50)
        mean = np.nanmean(list_values)
        std = np.nanstd(list_values)
        var = np.nanvar(list_values)
        #rms = np.nanmean(np.sqrt(list_values ** 2))
        return [mean, std, var] #[n5, n25, n75, n95, median, mean, std, var]


    X_Train_list = []
    x_test_list = []
    for i in X_Tr:#.values.tolist():
        a = calculate_statistics(i)
        X_Train_list.append(a)

    for i in x_t:#.values.tolist():
        a = calculate_statistics(i)
        x_test_list.append(a)

    def wavelet_transform(ecg_data, waveletname):
        list_features = []
        for signal in ecg_data:
            list_coef = pywt.wavedec(signal, waveletname)
            for coef in list_coef:
                list_features.append(coef)
        return np.asarray(list_features)


    X_Train_WT = wavelet_transform(X_Train_list, 'db4')
    x_test_wt = wavelet_transform(x_test_list, 'db4')
    return X_Train_WT, x_test_wt