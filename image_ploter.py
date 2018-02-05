import os
import glob
import csv
from scipy.io import wavfile
import matplotlib.pyplot as plt


def get_filenames(path, filetype=None):
    files = []
    if path[-1] != '/':
        path += '/'
    os.chdir(path)  # muda p/ pasta da base atual
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
def load_sound_files(file_paths):
    raw_sounds = []
    rates = []
    for f in file_paths:
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
    # name = i[1]+'.png'
    plt.savefig(title, dpi=100)

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
bases_Physio = [training_a, training_b, training_c, training_d, training_e, training_f]  # , validation]
path_Physio = '/home/bruno/Documentos/UFMA/mono/dataset/PhysioNet/'

label = ['Training B Normal', 'Btraining_murmur']

files = []
labels = []
file_paths = []

for base in bases_B:
    instances = []
    files.append(get_filenames(path=base, filetype='.wav'))
    labels.append(get_filenames(path=base, filetype='.hea'))
    for i in range(len(files[-1])):
        print base + '/' + files[-1][i]
        instance, rates = load_sound_files(base + '/' + files[-1][i])

files = get_filenames(btraning_mumur, '.wav')

# carrega arquivos de audio
for s in files:
    file_paths.append(btraning_mumur + '/' + s)
instance, rates = load_sound_files(file_paths)

print "Rotulo:", label[1]
print "Nome Arquivo:", files[0]
print "Dados:", instance[0], type(instance[0])
print "Frequencia (fs): ", rates[0]