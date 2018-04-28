import pandas as pd
import glob
import os

def main1():
    files = [f for f in os.listdir('.') if os.path.isfile(f)]

    merged = []

    for f in files:
        filename, ext = os.path.splitext(f)
        if ext == '.csv':
            read = pd.read_csv(f)
            merged.append(read)

    result = pd.concat(merged)

    result.to_csv('librosa_features_DatasetB_filtered.csv')


def main2():
    interesting_files = glob.glob("*.csv")

    header_saved = False
    with open('baseB_filtered_librosa_features.csv', 'wb') as fout:
        for filename in interesting_files:
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)


# main1()
main2()
