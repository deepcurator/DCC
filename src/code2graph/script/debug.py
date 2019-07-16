from pathlib import Path


path = Path("../../../../data_tf/09/NPRF-master/combined_triples.triples").resolve()

with open(path, 'r') as f:
    for line in f:
        split = line.split('\t')
        print(split)