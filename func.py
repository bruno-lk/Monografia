import glob
import os
import csv
import librosa
import pywt
# import pylab
import matplotlib.pyplot as plt
import numpy as np
# import heartBeat as hb
from scipy.io import wavfile
from scipy.signal import butter, lfilter, decimate
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
# from pyAudioAnalysis import audioSegmentation


# Gets all file names from a directory (extension may or may not be specified)
def get_filenames(path, filetype=None):
    files = []
    if path[-1] != '/':
        path += '/'
    os.chdir(path)  # muda p/ pasta da base atual # os.getcwd() para checar diretorio atual
    if filetype is None:
        for file in glob.glob('*'):
            files.append(file)
    else:
        for file in glob.glob('*' + filetype):
            files.append(file)
    return sorted(files)
    pass


def write_csv(filename, lista):  # raw=True
    if filename[-4:] != '.csv':
        filename += '.csv'
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in lista:
            writer.writerow(row)


# Read .wav files
def load_sound_files(file_paths):
    raw_sounds = []
    rates = []
    for f in file_paths:
        rate, data = wavfile.read(f)
        raw_sounds.append(data)
        rates.append(rate)
    return raw_sounds, rates


# plotagem dos sinais
def plot_imagens(data, title='', x='Tempo', y='Frequencia (Hz)'):
    plt.figure()
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid()
    plt.plot(data)
    # name = i[1]+'.png'
    # plt.savefig(name, dpi=100)


def extract_feature(X, sample_rate):  # file_name):
    # X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))  # Short-time Fourier transform
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)  # Mel-frequency cepstral coefficients
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)  # Compute a chromagram from a waveform or power spectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)  # Compute a mel-scaled spectrogram
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)  # Compute spectral contrast [R3333]
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)  # Computes the tonal centroid features (tonnetz), following the method of [R3737]
    return [mfccs, chroma, mel, contrast, tonnetz]


def pyAudioAnalysis_features(x, Fs):
    # [Fs, x] = audioBasicIO.readAudioFile(file_name)
    # stF = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.05 * Fs, 0.05 * Fs)
    mtF = audioFeatureExtraction.mtFeatureExtraction(x, Fs, 1 * Fs, 1 * Fs, 0.5 * Fs, 0.5 * Fs)
    return mtF
    # return [stF, mtF]

def wavelet_filtering(instance, th):
    decimateSignal = decimate(instance, 10)
    # transformada Wavelet Daubechies order 6
    coeffs = pywt.wavedec(decimateSignal, 'db6', level=4)
    cA4, cD4, cD3, cD2, cD1 = coeffs

    cD4 = pywt.threshold(cD4, th, mode='less')
    cD3 = pywt.threshold(cD3, th, mode='less')
    cD2 = pywt.threshold(cD2, th, mode='less')
    cD1 = pywt.threshold(cD1, th, mode='less')

    coeffs = cA4, cD4, cD3, cD2, cD1

    # reconstrucao do sinal
    recSignal = pywt.waverec(coeffs, 'db6')
    return recSignal


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def smoothTriangle(data, degree, dropVals=False):
    """performs moving triangle smoothing with a variable degree."""
    """note that if dropVals is False, output length will be identical
    to input length, but with copies of data at the flanking regions"""

    triangle = np.array(list(range(degree)) + [degree] + list(range(degree)[::-1])) + 1
    smoothed = []
    for i in range(degree, len(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed.append(sum(point) / sum(triangle))

    if dropVals:
        return smoothed

    smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed

    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])

    return smoothed


# lowpass
def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a


def butter_lowpass_filter(data, lowcut, fs, order=5):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

