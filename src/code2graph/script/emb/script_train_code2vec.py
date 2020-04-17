import sys, tqdm, pickle, random
from sklearn.model_selection import train_test_split

sys.path.append('../../')

from core.code2vec import *


def create_dataset_indexes(raw_data_path:Path, dataset_save_path:Path, filename):
    if (dataset_save_path / 'word_count.pkl').exists(): 
        print("Preprocess already done..")
        return
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
    if (dataset_save_path / "train.txt").exists():
        print("Train.txt exists.")
        return
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
                    for _ in range(3):
                        triple_ids = random.sample(triple_ids, max_contexts)

                        content = " ".join(triple_ids)
                        label_info = "|".join(label_ids)
                        # print(content)
                        # print(label_info)
                        data_functions.append((label_info, content))

    reduced_word_count = {}
    reduced_word2idx = {}
    reduced_idx2word = {}

    reduced_path_count = {}
    reduced_path2idx = {} 
    reduced_idx2path = {}

    reduced_target_count = {}
    reduced_target2idx = {}
    reduced_idx2target = {}
    
    for label, triples in data_functions:
        
        if idx2target[int(label)] not in reduced_target_count:
            reduced_target_count[idx2target[int(label)]] = 1
        else:
            reduced_target_count[idx2target[int(label)]] += 1

        for triple in triples.split(" "):
            splited_triple = triple.split('\t')
            # import pdb; pdb.set_trace()
            e1 = int(splited_triple[0])
            p = int(splited_triple[1])
            e2 = int(splited_triple[2])

            if idx2word[e1] in reduced_word_count:
                reduced_word_count[idx2word[e1]] += 1
            else:
                reduced_word_count[idx2word[e1]] = 1

            if idx2path[p] in reduced_path_count:
                reduced_path_count[idx2path[p]] += 1
            else:
                reduced_path_count[idx2path[p]] = 1
            
            if idx2word[e2] in reduced_word_count:
                reduced_word_count[idx2word[e2]] += 1
            else:
                reduced_word_count[idx2word[e2]] = 1

    reduced_word_count = {k: v for k, v in sorted(reduced_word_count.items(), key=lambda item: item[1], reverse=True)}
    reduced_word2idx = {v: k for k, v in enumerate(reduced_word_count.keys())}
    reduced_idx2word = {v: k for k, v in reduced_word2idx.items()}

    reduced_path_count = {k: v for k, v in sorted(reduced_path_count.items(), key=lambda item: item[1], reverse=True)}
    reduced_path2idx = {v: k for k, v in enumerate(reduced_path_count.keys())}
    reduced_idx2path = {v: k for k, v in reduced_path2idx.items()}

    reduced_target_count = {k: v for k, v in sorted(reduced_target_count.items(), key=lambda item: item[1], reverse=True)}
    reduced_target2idx = {v: k for k, v in enumerate(reduced_target_count.keys())}
    reduced_idx2target = {v: k for k, v in reduced_target2idx.items()}
    
    with open(str(dataset_save_path / 'reduced_word_count.pkl'), 'wb') as f:
        pickle.dump(reduced_word_count, f)
    with open(str(dataset_save_path / 'reduced_word2idx.pkl'), 'wb') as f:
        pickle.dump(reduced_word2idx, f)
    with open(str(dataset_save_path / 'reduced_idx2word.pkl'), 'wb') as f:
        pickle.dump(reduced_idx2word, f)

    with open(str(dataset_save_path / 'reduced_path_count.pkl'), 'wb') as f:
        pickle.dump(reduced_path_count, f)
    with open(str(dataset_save_path / 'reduced_path2idx.pkl'), 'wb') as f:
        pickle.dump(reduced_path2idx, f)
    with open(str(dataset_save_path / 'reduced_idx2path.pkl'), 'wb') as f:
        pickle.dump(reduced_idx2path, f)

    with open(str(dataset_save_path / 'reduced_target_count.pkl'), 'wb') as f:
        pickle.dump(reduced_target_count, f)
    with open(str(dataset_save_path / 'reduced_target2idx.pkl'), 'wb') as f:
        pickle.dump(reduced_target2idx, f)
    with open(str(dataset_save_path / 'reduced_idx2target.pkl'), 'wb') as f:
        pickle.dump(reduced_idx2target, f)

    reduced_data_functions = []

    for label, triples in data_functions:
        
        new_triples = []

        for triple in triples.split(' '):
            splited_triple = triple.split('\t')
            e1 = int(splited_triple[0])
            p = int(splited_triple[1])
            e2 = int(splited_triple[2])

            e1 = reduced_word2idx[idx2word[e1]]
            p = reduced_path2idx[idx2path[p]]
            e2 = reduced_word2idx[idx2word[e2]]

            new_triples.append("%s\t%s\t%s" % (e1, p, e2))
        
        reduced_data_functions.append((str(reduced_target2idx[idx2target[int(label)]]), ' '.join(new_triples)))

    train, test = train_test_split(reduced_data_functions, test_size=0.1, shuffle=True)

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