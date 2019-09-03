import gensim
import collections
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

        self.train_set = list(self.read_dataset(self.train_path))
        self.test_set = list(self.read_dataset(self.test_path), True)

    def read_dataset(self, fname, tokens_only=False):
        with open(fname, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                if tokens_only:
                    yield gensim.utils.simple_preprocess(line)
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

    def train_model(self, vector_size=50, min_count=2, epochs=40):
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size, min_count, epochs)
        self.model.build_vocab(train_set)
        self.model.train(train_set, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def infer_vector(self, function: str):
        assert(self.model)
        tokens = gensim.utils.simple_preprocess(function)
        return self.model.infer_vector(tokens)

    def get_most_similar_function(self, function: str):
        assert(self.model)
        vector = self.infer_vector(function)
        sims = self.model.docvecs.most_similar([vector], topn=len(model.docvecs))
        return sims[0]

if __name__ == "__main__":
    raw_data_path = Path("../functions").resolve()
    save_path = Path("../doc2vec").resolve()
    preprocess(raw_data_path, "functions.txt", save_path)
    
    d2v = Doc2Vec(save_path)
    d2v.train_model()
