import gensim
import collections
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

sys.path.append('../..')

from core.emb_doc2vec import Doc2Vec

def prepare_dataset(dataset_path: str, filename):
    
    raw_data_path = Path(dataset_path).resolve()
    dataset_save_path = Path("../doc2vec").resolve()
    dataset_save_path.mkdir(exist_ok=True)
    
    data_functions = [] 

    for code2vec_file in raw_data_path.rglob(filename):
        with open(code2vec_file, 'r') as file:
            for function_line in file:
                splited_function_line = function_line.split(" ")
                labels = splited_function_line[0]
                content = " ".join(splited_function_line[1:])
                data_functions.append((labels, content))

    train, test = train_test_split(data_functions, test_size=0.1, shuffle=True)

    with open(str(dataset_save_path / "train.txt"), 'w') as file:
        for labels, content in train:
            file.write(labels)
            file.write(" ")
            file.write(content)

    with open(str(dataset_save_path / "test.txt"), 'w') as file:
        for labels, content in test:
            file.write(labels)
            file.write(" ")
            file.write(content)

    return dataset_save_path

def run_doc2vec(dataset_path:str):
       
    dataset_save_path = prepare_dataset(dataset_path, "doc2vec.txt")

    d2v = Doc2Vec(dataset_save_path)
    d2v.train_model()
    d2v.save_model("d2v_model")

    d2v.load_model("d2v_model")
    d2v.dump_vectors()
    d2v.evaluate()
    
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    run_doc2vec('../../graphast_output')