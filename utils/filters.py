import scipy.signal as sg
import numpy as np
from scipy.signal import resample_poly

# -----------------------------
# Filtering helpers
# -----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = sg.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = sg.butter(order, cutoff / nyq, btype='low')
    return b, a


def apply_filter(x, fs, mode="ecg"):
    if len(x) < 16:
        return x
    if mode == "ecg":
        b, a = butter_bandpass(1, 40, fs)
    elif mode == "ppg":
        b, a = butter_bandpass(0.5, 5, fs)
    elif mode == "eda":
        b, a = butter_lowpass(1, fs)
    elif mode == "acc":
        b, a = butter_lowpass(10, fs)
    else:
        return x
    return sg.filtfilt(b, a, x)


def resample_to(x, orig_fs, target_fs=100):
    if orig_fs == target_fs:
        return x
    return resample_poly(x, target_fs, orig_fs)


def zscore(x):
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        return x - mean
    return (x - mean) / std
