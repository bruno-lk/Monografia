import os
import glob
import csv
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

for base in bases_B:
    instances = []
    files.append(func.get_filenames(path=base, filetype='.wav'))
    # labels.append(func.get_filenames(path=base, filetype='.hea'))
    for i in range(len(files[-1])):
        print('Done with: ' + files[-1][i])

    func.write_csv(path_B+base[-1], instances)
    print('Done with: ' + base + '\n')

