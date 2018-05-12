import os
import glob
import csv
import func
import matplotlib.pyplot as plt
import numpy as np
import librosa, librosa.display
from scipy.io import wavfile


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


def write_csv(filename, list, raw=True):
    if filename[-4:] != '.csv':
        filename += '.csv'
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in list:
            writer.writerow(row)


# Read .wav files
def load_sound_files(f):
    raw_sounds = []
    rates = []
    # for f in file_paths:
    rate, data = wavfile.read(f)
    raw_sounds.append(data)
    rates.append(rate)
    return raw_sounds, rates


# plotagem dos sinais
def plot_imagens(data, title=''):
    plt.figure()
    plt.title(title)
    plt.xlabel('Tempo')
    plt.ylabel('Freqencia (Hz)')
    plt.grid()
    plt.plot(data)
    plt.savefig(title + '.png', dpi=100)
    plt.clf()


# plotagem dos spectogramas dos sinais
def plot_spectogram(y, title):
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    # plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram - ' + title)
    plt.savefig(title + '.png', dpi=100)
    plt.clf()

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

label = ['Training B Normal', 'Btraining_murmur']

files = []
labels = []
file_paths = []

for base in bases_B:
    instances = []
    files.append(get_filenames(path=base, filetype='.wav'))
    # labels.append(get_filenames(path=base, filetype='.hea'))

    # a maneira mais correta de fazer seria dividir em dois loops
    # assim o entendimento poderia ficar mais facil (?)
    for i in range(len(files[-1])):
        # instance, rate = load_sound_files(base + '/' + files[-1][i])
        instance, rate = librosa.load(base + '/' + files[-1][i])
        rec_signal = func.wavelet_filtering(instance, th=288)
        plot_imagens(rec_signal, files[-1][i] + " filtered waveplot")
        # plot_spectogram(rec_signal, files[-1][i] + ' filtered')
        print (files[-1][i] + ' done!')
    print('Done with: ' + base + '\n')

# print (base + '/' + files[-1][0])

# x, sr = librosa.load(base + '/' + files[-1][0])
# librosa.display.waveplot(x, sr=sr)
# plt.show()


# files = get_filenames(btraning_mumur, '.wav')
#
# # carrega arquivos de audio
# for s in files:
#     file_paths.append(btraning_mumur + '/' + s)
# instance, rates = load_sound_files(file_paths)
#
# print "Rotulo:", label[1]
# print "Nome Arquivo:", files[0]
# print "Dados:", instance[0], type(instance[0])
# print "Frequencia (fs): ", rates[0]