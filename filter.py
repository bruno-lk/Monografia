import os
# import glob
# import csv
import pywt
import func
# import gc
# import numpy as np
# import matplotlib.pyplot as plt
from scipy import signal
# from pyAudioAnalysis import audioFeatureExtraction as aF
# import heartbeat as hb
# import pandas as pd
from scipy.io import wavfile
# from math import isnan

'''
A ideia e ler cada faia de audio e filtrar:
    carregar arquivo na memoria
    fazer a filtragem com wavelet
    escrever o vetor no .csv
limpar a memoria para testar se nao trava
'''

# dataset PASCAL B
btraning_normal = '/home/bruno/Documentos/UFMA/mono/dataset/B/Training B Normal'
btraning_mumur = '/home/bruno/Documentos/UFMA/mono/dataset/B/Btraining_murmur'
btraining_extrastole = '/home/bruno/Documentos/UFMA/mono/dataset/B/Btraining_extrastole'
bases_B = [btraning_normal, btraining_extrastole, btraning_mumur]
path_B = '/home/bruno/Documentos/UFMA/mono/dataset/B/'

# Dataset PhysioNet
training_a = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/training/training-a'
training_b = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/training/training-b'
training_c = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/training/training-c'
training_d = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/training/training-d'
training_e = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/training/training-e'
training_f = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/training/training-a'
validation = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/validation'
# bases_Physio = [, , ]  # , validation]
bases_Physioab = [training_a, training_b]
bases_Physiocd = [training_c, training_d]
bases_Physioef = [training_e, training_f]
path_Physio = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/'

# dataset Michigan_Heart_Sounds
michigan = '/home/bruno/Documentos/UFMA/mono/dataset/Michigan_Heart_Sounds'
base_michigan = [michigan]

files = []
labels = []
file_paths = []

# fs = 4000.0
# hrw = 0.05
# lowcut = 195.0
# highcut = 882.0
th = 195.0

# Wavelet Daubechies order 6 (ortogonal)
# wavelet = pywt.Wavelet('db6')
# (phi, psi, x) = wavelet.wavefun()

for base in bases_Physiocd:
    instances = []
    files.append(func.get_filenames(path=base, filetype='.wav'))
    # labels.append(func.get_filenames(path=base, filetype='.hea'))

    for i in range(len(files[-1])):
        # instance, rate = func.load_sound_files(base + '/' + files[-1][i])
        rate, instance = wavfile.read(base + '/' + files[-1][i])

        # filtragem e decomposicao
        decimateSignal = signal.decimate(instance, 10)
        coeffs = pywt.wavedec(decimateSignal, 'db6', level=4)  # transformada Wavelet Daubechies order 6
        cA4, cD4, cD3, cD2, cD1 = coeffs

        cD4 = pywt.threshold(cD4, th, mode='less')
        cD3 = pywt.threshold(cD3, th, mode='less')
        cD2 = pywt.threshold(cD2, th, mode='less')
        cD1 = pywt.threshold(cD1, th, mode='less')

        coeffs = cA4, cD4, cD3, cD2, cD1

        # reconstrucao do sinal
        recSignal = pywt.waverec(coeffs, 'db6')

        # adicao do valor no vetor
        instances.append(recSignal)
        print('Done with: ' + files[-1][i])

    # # func.write_csv(path_B+base[-1], instances)
    dir = os.path.basename(base)
    func.write_csv(dir + '_filter', instances)
    print('Done with: ' + base + '\n')

