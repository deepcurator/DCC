import gensim
import collections
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def save_processed_data(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write(line)


def preprocess(raw_data_path: Path, filename, save_path: Path):
    
    functions = [] 

    for file in raw_data_path.rglob(filename):
        with open(file, 'r') as f:
            for line in f:
                functions.append(line)

    train, test = train_test_split(functions, test_size=0.1)
    save_processed_data(train, str(save_path / "train.txt"))
    save_processed_data(test, str(save_path / "test.txt"))


class Doc2Vec:

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).resolve()
        self.train_path = data_dir / "train.txt"
        self.test_path = data_dir / "test.txt"

        self.train_corpus = list(self.read_dataset(self.train_path))
        self.test_corpus = list(self.read_dataset(self.test_path, True))

    def read_dataset(self, fname, tokens_only=False):
        with open(fname, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                if tokens_only:
                    yield gensim.utils.simple_preprocess(line)
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

    def train_model(self, vector_size=50, window=2, min_count=2, epochs=40, workers=4):
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, window=window, 
                                                   min_count=min_count, epochs=epochs,
                                                   workers=workers)
        self.model.build_vocab(self.train_corpus)
        self.model.train(self.train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def get_learned_vector(self, tag):
        return self.model.docvecs[tag]

    def dump_vectors(self):
        file_path = self.data_dir / "doc2vec_vectors.txt"
        self.model.docvecs.save_word2vec_format(str(file_path))

    def infer_vector(self, function: str):
        assert(self.model)
        tokens = gensim.utils.simple_preprocess(function)
        return self.model.infer_vector(tokens)

    def get_most_similar_function(self, function: str):
        assert(self.model)
        vector = self.infer_vector(function)
        sims = self.model.docvecs.most_similar([vector], topn=len(model.docvecs))
        return ' '.join(self.train_corpus[sims[0][0]].words)

    def save_model(self, fname):
        save_path = self.data_dir / fname
        self.model.save(str(save_path))

    def load_model(self, fname):
        load_path = self.data_dir / fname
        self.model = gensim.models.doc2vec.Doc2Vec.load(str(load_path))

if __name__ == "__main__":

    raw_data_path = Path("../graphast_output").resolve()
    save_path = Path("../doc2vec").resolve()
    save_path.mkdir(exist_ok=True)
    # preprocess(raw_data_path, "functions.txt", save_path)
    
    d2v = Doc2Vec(save_path)
    # d2v.train_model()
    # d2v.save_model("d2v_model")
    # d2v.load_model("d2v_model")
    # import pdb; pdb.set_trace()