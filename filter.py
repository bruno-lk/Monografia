import func
import librosa
import os
from pyAudioAnalysis import audioBasicIO
# import glob
# import csv
# import pywt
# import gc
import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal
from pyAudioAnalysis import audioFeatureExtraction
# import heartbeat as hb
# import pandas as pd
# from math import isnan
# from scipy.io import wavfile

# dataset PASCAL A
atraning_normal = '/home/bruno/Documentos/UFMA/mono/dataset/A/Atraining_normal'
atraning_mumur = '/home/bruno/Documentos/UFMA/mono/dataset/A/Atraining_murmur'
atraining_extrahls = '/home/bruno/Documentos/UFMA/mono/dataset/A/Atraining_extrahls'
atraining_artifact = '/home/bruno/Documentos/UFMA/mono/dataset/A/Atraining_artifact'
bases_A = [atraning_normal, atraining_extrahls, atraning_mumur, atraining_artifact]
path_A = '/home/bruno/Documentos/UFMA/mono/dataset/A/'

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
training_f = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/training/training-f'
validation = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/validation'
# bases_Physio = [training_a, training_b, training_c, training_d, training_e, training_f]
bases_Physioab = [training_a, training_b]
bases_Physiocd = [training_c, training_d]
bases_Physioef = [training_e, training_f]
path_Physio = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/training/'

# dataset Michigan_Heart_Sounds
michigan = '/home/bruno/Documentos/UFMA/mono/dataset/Michigan_Heart_Sounds'
base_michigan = [michigan]


file_paths = []

# fs = 4000.0
# hrw = 0.05
# lowcut = 195.0
# highcut = 882.0
th = 288.0
time_frame = 20
over_frame = time_frame/2
frame_length = 1024
hop = 512

# Wavelet Daubechies order 6 (ortogonal)
# wavelet = pywt.Wavelet('db6')
# (phi, psi, x) = wavelet.wavefun()


def test_pyAudioAnalysis():
    [Fs, x] = audioBasicIO.readAudioFile("diarizationExample.wav")
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
    plt.subplot(2, 1, 1)
    plt.plot(F[0, :])
    plt.xlabel('Frame no')
    plt.ylabel('ZCR')
    plt.subplot(2, 1, 2)
    plt.plot(F[1, :])
    plt.xlabel('Frame no')
    plt.ylabel('Energy')
    plt.show()


def gerador_de_caracteriticas():
    all_features = []

    for base in bases_B:
        features_list = []
        files.append(func.get_filenames(path=base, filetype='.wav'))

        for i in range(len(files[-1])):
            instance, rate = librosa.load(base + '/' + files[-1][i])
            features = func.extract_feature(instance, rate)

            # rotulacao
            if base == btraning_normal:
                features.append("Normal")
            elif base == btraning_mumur:
                features.append("Murmur")
            elif base == btraining_extrastole:
                features.append("Extrastole")

            features_list.append(features)
            all_features.append(features)
            print('Done with: ' + files[-1][i] + " - " + features[6])

        # escrita em arquivo CSV
        dir_db = os.path.basename(base)
        func.write_csv('librosa_features_of_' + dir_db, features_list)
        print('Done with: ' + base + '\n')

    os.chdir(path_B)
    func.write_csv("BaseB_librosa_features", all_features)


def gerador_de_audio():
    files = []

    for base in bases_B:
        instances_list = []
        rate_list = []

        files.append(func.get_filenames(path=base, filetype='.wav'))

        for i in range(len(files[-1])):
            instance, rate = librosa.load(base + '/' + files[-1][i])

            # filtragem e decomposicao
            rec_signal = func.wavelet_filtering(instance, th)
            instances_list.append(rec_signal)
            rate_list.append(rate)
            print('Done with: ' + files[-1][i])

        print('Done with: ' + base + '\n')

        # mudanca p/ novo diretorio
        new_path = base + "/filtered"
        os.chdir(new_path)

        for i in range(len(files[-1])):
            # geracao de arquivo
            # maxv = np.iinfo(np.int16).max
            # librosa.output.write_wav(
            #     files[-1][i] + "_filtered_int16.wav", (instances_list[i] * maxv).astype(np.int16), rate_list[i]
            # )
            librosa.output.write_wav(files[-1][i] + "_filtered.wav", instances_list[i], rate_list[i])
            print('Done with audio: ' + files[-1][i])

        print('Done with: ' + base + '\n')


def main_a():
    files = []
    all_features = []

    for base in bases_A:
        instances_list = []
        features_list = []
        filtered_instances_list = []
        files.append(func.get_filenames(path=base, filetype='.wav'))  # gera array com nome dos audios

        for i in range(len(files[-1])):
            # instance, rate = func.load_sound_files(base + '/' + files[-1][i])
            # rate, instance = wavfile.read(base + '/' + files[-1][i])
            instance, rate = librosa.load(base + '/' + files[-1][i])
            # [rate, instance] = audioBasicIO.readAudioFile(base + '/' + files[-1][i])

            instances_list.append(instance)

            # filtragem e decomposicao
            rec_signal = func.wavelet_filtering(instance, th)
            filtered_instances_list.append(rec_signal)

            # extracao de caracteriticas
            features = func.extract_feature(instance, rate)  # librosa
            # features = audioFeatureExtraction.stFeatureExtraction(instance, rate, 0.5 * rate, 0.25 * rate)
            # features = audioFeatureExtraction.mtFeatureExtraction(
            #     instance, rate, 0.5 * rate, 0.5 * rate,0.25 * rate, 0.25 * rate
            # )

            # rotulacao
            if base == atraning_normal:
                features.append("Normal")
            elif base == atraning_mumur:
                features.append("Murmur")
            elif base == atraining_extrahls:
                features.append("Extra Heart Sound")
            elif base == atraining_artifact:
                features.append("Artifact")

            features_list.append(features)
            all_features.append(features)
            print('Done with: ' + files[-1][i])

        # escritas no CVS
        dir_db = os.path.basename(base)
        func.write_csv('librosa_filtered_features_of_' + dir_db, features_list)
        print('Done with: ' + base + '\n')

    os.chdir(path_A)
    func.write_csv("BaseA_librosa_filtered_features_of_", all_features)
    print 'Done!'


def main_b():
    files = []
    all_features = []

    for base in bases_B:
        instances_list = []
        features_list = []
        filtered_instances_list = []
        files.append(func.get_filenames(path=base, filetype='.wav'))  # gera array com nome dos audios

        for i in range(len(files[-1])):
            # instance, rate = func.load_sound_files(base + '/' + files[-1][i])
            # rate, instance = wavfile.read(base + '/' + files[-1][i])
            instance, rate = librosa.load(base + '/' + files[-1][i])
            # [rate, instance] = audioBasicIO.readAudioFile(base + '/' + files[-1][i])

            instances_list.append(instance)

            # filtragem e decomposicao
            rec_signal = func.wavelet_filtering(instance, th)
            filtered_instances_list.append(rec_signal)

            # extracao de caracteriticas
            features = func.extract_feature(rec_signal, rate)  # librosa
            # features = audioFeatureExtraction.stFeatureExtraction(instance, rate, 0.5 * rate, 0.25 * rate)
            # features = audioFeatureExtraction.mtFeatureExtraction(
            #     instance, rate, 0.5 * rate, 0.5 * rate,0.25 * rate, 0.25 * rate
            # )

            # rotulacao
            if base == btraning_normal:
                features.append("Normal")
            elif base == btraning_mumur:
                features.append("Murmur")
            elif base == btraining_extrastole:
                features.append("Extrastole")

            features_list.append(features)
            all_features.append(features)
            print('Done with: ' + files[-1][i])

    #     # escritas no CVS
    #     dir_db = os.path.basename(base)
    #     func.write_csv('librosa_filtered_features_of_' + dir_db, features_list)
    #     print('Done with: ' + base + '\n')
    #
    # # escritas no CVS geral
    # os.chdir(path_B)
    # func.write_csv("BaseB_librosa_filtered_features_of_", all_features)
    print 'Done!'


def main_pn():
    files = []
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


# gerador_de_caracteriticas()
# main_a()
main_b()
# main_pn()
# test_pyAudioAnalysis()
# gerador_de_audio()
