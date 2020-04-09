import sys, tqdm, pickle, random
from sklearn.model_selection import train_test_split

sys.path.append('../../')

from core.code2vec import *


def create_dataset_indexes(raw_data_path:Path, dataset_save_path:Path, filename):

    word_count = {}
    word2idx = {}
    idx2word = {}

    path_count = {}
    path2idx = {} 
    idx2path = {}

    target_count = {}
    target2idx = {}
    idx2target = {}

    data_functions = [] 

    for code2vec_file in raw_data_path.rglob(filename):
        with open(code2vec_file, 'r') as file:
            for function_line in file:
                splited_function_line = function_line.split(" ")
                label = splited_function_line[0].split('.')[-1]

                if label not in target_count:
                    target_count[label]=1
                else:
                    target_count[label]+=1

                triples = splited_function_line[1:]

                for triple in triples:
                    splited_triple = triple.split('\t')

                    if len(splited_triple) != 3: 
                        continue

                    e1, p, e2 = splited_triple[0], splited_triple[1], splited_triple[2]
                    
                    if e1 not in word_count:
                        word_count[e1]=1
                    else:
                        word_count[e1]+=1

                    if e2 not in word_count:
                        word_count[e2]=1
                    else:
                        word_count[e2]+=1

                    if p not in path_count:
                        path_count[p]=1
                    else:
                        path_count[p]+=1
    
    word_count = {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)}
    word2idx = {v: k for k, v in enumerate(word_count.keys())}
    idx2word = {v: k for k, v in word2idx.items()}

    path_count = {k: v for k, v in sorted(path_count.items(), key=lambda item: item[1], reverse=True)}
    path2idx = {v: k for k, v in enumerate(path_count.keys())}
    idx2path = {v: k for k, v in path2idx.items()}

    target_count = {k: v for k, v in sorted(target_count.items(), key=lambda item: item[1], reverse=True)}
    target2idx = {v: k for k, v in enumerate(target_count.keys())}
    idx2target = {v: k for k, v in target2idx.items()}

    with open(str(dataset_save_path / 'word_count.pkl'), 'wb') as f:
        pickle.dump(word_count, f)
    with open(str(dataset_save_path / 'word2idx.pkl'), 'wb') as f:
        pickle.dump(word2idx, f)
    with open(str(dataset_save_path / 'idx2word.pkl'), 'wb') as f:
        pickle.dump(idx2word, f)

    with open(str(dataset_save_path / 'path_count.pkl'), 'wb') as f:
        pickle.dump(path_count, f)
    with open(str(dataset_save_path / 'path2idx.pkl'), 'wb') as f:
        pickle.dump(path2idx, f)
    with open(str(dataset_save_path / 'idx2path.pkl'), 'wb') as f:
        pickle.dump(idx2path, f)

    with open(str(dataset_save_path / 'target_count.pkl'), 'wb') as f:
        pickle.dump(target_count, f)
    with open(str(dataset_save_path / 'target2idx.pkl'), 'wb') as f:
        pickle.dump(target2idx, f)
    with open(str(dataset_save_path / 'idx2target.pkl'), 'wb') as f:
        pickle.dump(idx2target, f)

    return "Success"


def preprocess_dataset(raw_data_path, dataset_save_path:Path, filename):
    # maximum number of contexts to keep for each function 
    max_contexts = 200 

    max_words = 10000000
    max_paths = 10000000

    word_count = {}
    word2idx = {}
    idx2word = {}

    path_count = {}
    path2idx = {} 
    idx2path = {}

    target_count = {}
    target2idx = {}
    idx2target = {}

    with open(str(dataset_save_path / 'word_count.pkl'), 'rb') as f:
        word_count = pickle.load(f)
    with open(str(dataset_save_path / 'word2idx.pkl'), 'rb') as f:
        word2idx = pickle.load(f)
    with open(str(dataset_save_path / 'idx2word.pkl'), 'rb') as f:
        idx2word = pickle.load(f)

    with open(str(dataset_save_path / 'path_count.pkl'), 'rb') as f:
        path_count = pickle.load(f)
    with open(str(dataset_save_path / 'path2idx.pkl'), 'rb') as f:
        path2idx = pickle.load(f)
    with open(str(dataset_save_path / 'idx2path.pkl'), 'rb') as f:
        idx2path = pickle.load(f)

    with open(str(dataset_save_path / 'target_count.pkl'), 'rb') as f:
        target_count = pickle.load(f)
    with open(str(dataset_save_path / 'target2idx.pkl'), 'rb') as f:
        target2idx = pickle.load(f)
    with open(str(dataset_save_path / 'idx2target.pkl'), 'rb') as f:
        idx2target = pickle.load(f)

    data_functions = [] 

    for code2vec_file in raw_data_path.rglob(filename):
        with open(code2vec_file, 'r') as file:
            for function_line in file:
                splited_function_line = function_line.split(" ")
                label = splited_function_line[0].split('.')[-1]
                label_ids = []
                
                label_ids.append(str(target2idx[label]))

                triples = splited_function_line[1:]
                triple_ids = []

                counter = 0 
                for triple in triples:
                    splited_triple = triple.split('\t')
                    
                    if len(splited_triple) != 3: 
                        counter += 1
                        continue
                    e1, p, e2 = splited_triple[0], splited_triple[1], splited_triple[2]
                    # print(word2idx[e1], path2idx[p], word2idx[e2])
                    
                    triple_ids.append("%s\t%s\t%s"%(word2idx[e1], path2idx[p], word2idx[e2]))
                
                num_contexts = len(triples)-counter
                if num_contexts > max_contexts:
                    triple_ids = random.sample(triple_ids, max_contexts)

                content = " ".join(triple_ids)
                label_info = "|".join(label_ids)
                # print(content)
                # print(label_info)
                data_functions.append((label_info, content))

    train, test = train_test_split(data_functions, test_size=0.1, shuffle=True)

    with open(str(dataset_save_path / "train.txt"), 'w') as file:
        for labels, content in train:
            file.write(labels)
            file.write(" ")
            file.write(content)
            file.write("\n")

    with open(str(dataset_save_path / "test.txt"), 'w') as file:
        for labels, content in test:
            file.write(labels)
            file.write(" ")
            file.write(content)
            file.write("\n")


def prepare_dataset(dataset_path: str, filename):
    
    raw_data_path = Path(dataset_path).resolve()
    dataset_save_path = Path("../code2vec").resolve()
    dataset_save_path.mkdir(exist_ok=True)

    create_dataset_indexes(raw_data_path, dataset_save_path, filename) # create dictionaries. 
    preprocess_dataset(raw_data_path, dataset_save_path, filename) # preprocessing the dataset. 

    return dataset_save_path

def run_code2vec(dataset_path:str):

    dataset_save_path = prepare_dataset(dataset_path, "code2vec.txt")

    trainer = Trainer(dataset_save_path)
    trainer.train_model()
    trainer.evaluate_model()

    # code.interact(local=locals())

if __name__ == "__main__":
    run_code2vec('../../graphast_output')