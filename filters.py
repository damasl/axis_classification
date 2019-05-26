import numpy as np
from scipy import signal

def erosion(signal, element):
    signal_len = len(signal)
    element_len = len(element)
    result = np.zeros((signal_len,))
    window =  (element_len - 1)//2
    for n in range(signal_len):
        left_window = min(n, window)
        right_window = min(signal_len-n-1, window)
        try:
            result[n] = np.min(signal[n-left_window:n+right_window+1] - element[window-left_window:window+right_window+1])
        except:
            print(n, left_window, right_window, window)
    return result

def dilation(signal, element):
    signal_len = len(signal)
    element_len = len(element)
    result = np.zeros((signal_len,))
    window =  (element_len - 1)//2
    for n in range(signal_len):
        left_window = min(n, window)
        right_window = min(signal_len-n-1, window)
        try:
            result[n] = np.max(signal[n-left_window:n+right_window+1][::-1] - element[window-left_window:window+right_window+1])
        except:
            print(n, left_window, right_window, window)
    return result

def opening(signal, element):
    return dilation(erosion(signal, element), element)

def closing(signal, element):
    return erosion(dilation(signal, element), element)

def step(signal, element):
    left = opening(closing(signal, element), element)
    right = closing(opening(signal, element), element)
    return (left+right)/2

def bandpass(data, fs, fc_low=5, fc_high=20):
    b, a = signal.butter(2, 
                         [float(fc_low) * 2 / fs,
                          float(fc_high) * 2 / fs],
                         'pass')
    return signal.filtfilt(b, a, data, axis=0)

def bandstop(data, fs, fc_stop = 60, fc_window=5):
    b, a = signal.butter(2, [float(fc_stop - fc_window) * 2 / fs,
                             float(fc_stop + fc_window) * 2 / fs], 'bandstop')
    return signal.filtfilt(b, a, data,
                                 axis=0)


def bandpass_filter(signal, params):
    fs = params.get('fs', 500)
    low = params.get('low', 0.5)
    high = params.get('high', 150)
    return bandpass(signal, fs, low, high)

def frequenncy_filter(signal, params):
    fs = params.get('fs', 500)
    stop = params.get('stop', 50)
    window = params.get('window', 5)
    low = params.get('low', 0.5)
    high = params.get('high', 150)
    return bandstop(bandpass(signal, fs, low, high), fs, stop, window)

def chu_filter(signal, params):
    se1 = params.get("se1", np.zeros(55))
    se2 = params.get('se2', np.zeros(135))
    return signal - (step(step(signal, se1), se2))
    