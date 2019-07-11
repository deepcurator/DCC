from pathlib import Path
from random import shuffle
import math
import glob

def save_to_file(data, name, path):
    file_path = Path(path) / name
    file_path = file_path.resolve()
    with open(file_path, "w") as f:
        for line in data:
            f.write(line + '\n')

if __name__ == "__main__":
    dataset = []
    path = Path("../rdf_triples")
    path = Path(path.resolve())
    for file in path.rglob("combined_triples.triples"):
        with open(file, "r") as f:
            for line in f:
                split = line.split('\t')
                if len(split) == 3:
                    dataset.append(line)
                else:
                    print(file)
                    print(line)
                
    shuffle(dataset)
    ten_percent = math.ceil(len(dataset) / 10)
    # import pdb; pdb.set_trace()
    train = dataset[2*ten_percent:]
    test = dataset[:ten_percent]
    valid = dataset[ten_percent:2*ten_percent]

    save_path = Path("../dataset")
    save_path = save_path.resolve()
    Path(save_path).mkdir(exist_ok=True)

    save_to_file(train, "train.txt", save_path)
    save_to_file(test, "test.txt", save_path)
    save_to_file(valid, "valid.txt", save_path)