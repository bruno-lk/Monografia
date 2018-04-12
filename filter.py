import func
import librosa
import os
# import glob
# import csv
# import pywt
# import gc
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# from pyAudioAnalysis import audioFeatureExtraction as aF
# import heartbeat as hb
# import pandas as pd
# from math import isnan
# from scipy.io import wavfile

# dataset PASCAL B
btraning_normal      = '/home/bruno/Documentos/UFMA/mono/dataset/B/Training B Normal'
btraning_mumur       = '/home/bruno/Documentos/UFMA/mono/dataset/B/Btraining_murmur'
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
# bases_Physio = [training_a, training_b, training_c, training_d, training_e, training_f]
bases_Physioab = [training_a, training_b]
bases_Physiocd = [training_c, training_d]
bases_Physioef = [training_e, training_f]
path_Physio = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/training/'

# dataset Michigan_Heart_Sounds
michigan = '/home/bruno/Documentos/UFMA/mono/dataset/Michigan_Heart_Sounds'
base_michigan = [michigan]

files = []
file_paths = []

# fs = 4000.0
# hrw = 0.05
# lowcut = 195.0
# highcut = 882.0
th = 195.0
time_frame = 20
over_frame = time_frame/2
frame_length = 1024
hop = 512

# Wavelet Daubechies order 6 (ortogonal)
# wavelet = pywt.Wavelet('db6')
# (phi, psi, x) = wavelet.wavefun()


def main_b():
    for base in bases_B:
        instances_list = []
        features_list = []
        files.append(func.get_filenames(path=base, filetype='.wav'))
        # labels.append(func.get_filenames(path=base, filetype='.hea'))

        for i in range(len(files[-1])):
            # instance, rate = func.load_sound_files(base + '/' + files[-1][i])
            # rate, instance = wavfile.read(base + '/' + files[-1][i])
            instance, rate = librosa.load(base + '/' + files[-1][i])

            # filtragem e decomposicao
            rec_signal = func.wavelet_filtering(instance, th)
            instances_list.append(rec_signal)

            # extracao de caracteriticas
            features = func.extract_feature(rec_signal, rate)

            # rotulacao
            if base == btraning_normal:
                features.append("Normal")
            else:
                features.append("Anormal")

            # features = func.pyAudioAnalysis_features(recSignal, rate)
            features_list.append(features)

            print('Done with: ' + files[-1][i])

        # func.write_csv(path_Physio+base[-1], features_list)
        dir_db = os.path.basename(base)
        func.write_csv('librosa_features_of_' + dir_db + '_filtered', features_list)
        print('Done with: ' + base + '\n')
    print 'Done!'


def main_pn():
    labels = []

    for base in bases_Physioef:
        instances_list = []
        features_list = []
        files.append(func.get_filenames(path=base, filetype='.wav'))
        labels.append(func.get_filenames(path=base, filetype='.hea'))

        for i in range(len(files[-1])):
            # instance, rate = func.load_sound_files(base + '/' + files[-1][i])
            # rate, instance = wavfile.read(base + '/' + files[-1][i])
            instance, rate = librosa.load(base + '/' + files[-1][i])

            # filtragem e decomposicao
            rec_signal = func.wavelet_filtering(instance, th)
            instances_list.append(rec_signal)

            # extracao de caracteriticas
            features = func.extract_feature(rec_signal, rate)

            # rotulacao physionet
            label_object = open(base + '/' + labels[-1][i], 'r')
            label = label_object.read().split('\n')[-2]
            features.append(label)

            # features = func.pyAudioAnalysis_features(recSignal, rate)
            features_list.append(features)

            print('Done with: ' + files[-1][i])

        func.write_csv(path_Physio+base[-1] + '_filtered', features_list)
        # dir_db = os.path.basename(base)
        # func.write_csv('librosa_features_of_' + dir_db, features_list)
        print('Done with: ' + base + '\n')
    print 'Done!'


def clear_files(filename):
    with open(filename, mode='r') as file:
        data = file.readlines()
        wholedata = []
        for line in data:
            wholedata += list(
                    filter(None, str.join(',', list(
                        filter(None, line.split(' '))
                    )
                                          )
                           .replace('"', '')
                           .replace('[', '')
                           .replace(']', '')
                           .replace('\n', '')
                           .replace('#', '')
                           .split(',')))
        minlist = []
        wholist = []
        for i in range(len(wholedata)):
            minlist.append(wholedata[i])
            if wholedata[i] == 'Normal' or wholedata[i] == 'Anormal':
                wholist.append(minlist)
                minlist = []
        print(wholist)
        print(len(wholist))
        func.write_csv(filename+'_1', wholist)
    pass


# main_b()
main_pn()
# clear_files(path_Physio + 'f')
