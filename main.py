import pywt
import func
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# from pyAudioAnalysis import audioFeatureExtraction as aF
# import heartbeat as hb
# import pandas as pd
# from scipy.io import wavfile
# from math import isnan

label = ['Training B Normal', 'Btraining_murmur']
baseb = '/home/bruno/Documentos/Python/preProcessamento/dataset/B/Training B Normal'
mumur = '/home/bruno/Documentos/Python/preProcessamento/dataset/B/Btraining_murmur'
path = '/home/bruno/Documentos/Python/preProcessamento/dataset/B/'

fs = 4000.0
hrw = 0.05
lowcut = 195.0
highcut = 882.0
th = 195.0

def main():
    # files = []
    # instance = []
    labels = []
    file_paths = []

    files = func.get_filenames(baseb, '.wav')

    # carrega arquivos de audio
    for s in files:
        file_paths.append(baseb + '/' + s)
    instance, rates = func.load_sound_files(file_paths)

    # ecg = pywt.data.ecg() # dataset do pywavelets

    # Wavelet Daubechies order 6 (ortogonal)
    wavelet = pywt.Wavelet('db6')
    (phi, psi, x) = wavelet.wavefun()
    # wavelet = signal.daub(6)

    print "Rotulo:", label[1]
    print "Nome Arquivo:", files[0]
    # print "Dados:", instance[0], type(instance[0])
    print "Frequencia (fs): ", rates[0]
    print "\n", wavelet


    # filtragem e decomposicao
    decimateSignal = signal.decimate(instance[0], 10)  # decomposicao de sinal
    # y = func.butter_lowpass_filter(decimateSignal, lowcut, fs, order=6)  # passa baixa
    coeffs = pywt.wavedec(decimateSignal, 'db6', level=4)  # transformada wavelet
    cA4, cD4, cD3, cD2, cD1 = coeffs
    # print "cD4:", cD4
    # print "cD3:", cD3
    # print "cD2:", cD2
    # print "cD1:", cD1

    cD4 = pywt.threshold(cD4, th, mode='less')
    cD3 = pywt.threshold(cD3, th, mode='less')
    cD2 = pywt.threshold(cD2, th, mode='less')
    cD1 = pywt.threshold(cD1, th, mode='less')

    coeffs = cA4, cD4, cD3, cD2, cD1

    # new_coeffs = []
    # aux = []
    # for cd in coeffs[1:]:
    #     for d in cd:
    #         if 288.0 >= d > 0.0:
    #             aux = np.append(aux, d)
    # new_coeffs.append(aux)


    # reconstrucao do sinal
    recSignal = pywt.waverec(coeffs, 'db6')  # recuperacao de sinal

    # verifica se sinais sao iguais
    if np.array_equal(recSignal, instance[0]):
        print('! sinais iguais !')

    smooth = signal.savgol_filter(recSignal, 5, 2)
    # print smooth

    # z = np.polyfit(smooth, , 3)
    # print z

    std = np.std(smooth)
    print "std", std
    helper = []
    for i, x in enumerate(smooth):
        if x > std:
            helper.append(i)

    print helper

    # # Spectrogram
    # f, Pwelch_spec = signal.welch(recSignal, fs, scaling='spectrum')
    # plt.semilogy(f, Pwelch_spec)
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD')
    # plt.grid()
    # plt.show()
    #
    # f = pywt.threshold(f, 195, mode='less')
    # plt.semilogy(f, Pwelch_spec)
    # plt.show()
    #
    # spectro = aF.stSpectogram(recSignal, rates[0], 0.05*rates[0], 0.025*rates[0], PLOT=True)
    # print spectro

    # Segmentacao

    # TODO segmentacao


    # Plotagem dos sinais
    func.plot_imagens(psi, 'Wavelet Daubechies de Ordem 6')# wavelet

    func.plot_imagens(instance[0], 'sinal nao filtrado')
    # func.plot_imagens(decimateSignal, 'sinal reduzido - fator 10')

    func.plot_imagens(recSignal, 'sinal reconstruido')
    # func.plot_imagens(recSignal2, 'sinal reconstruido - upcoef')

    func.plot_imagens(smooth, 'suavizacao triangular')
    # func.plot_imagens(z,'polyfit')

    plt.show()

main()
