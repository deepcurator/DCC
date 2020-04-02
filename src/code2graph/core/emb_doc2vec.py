import gensim
import collections
import numpy as np
from pathlib import Path


class Doc2Vec:
    ''' 
        doc2vec is a method that learns paragraph and document embeddings. 
        It is proposed by Mikilov and Le in 2014. 
        This class includes a Gensim implementation of doc2vec. 
    '''
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).resolve()
        self.train_path = data_dir / "train.txt"
        self.test_path = data_dir / "test.txt"

        self.train_corpus = list(self.read_dataset(self.train_path))
        self.test_corpus = list(self.read_dataset(self.test_path, True))

    def read_dataset(self, fname, tokens_only=False):
        with open(fname, encoding="iso-8859-1") as f:
            for line in f:
                splited_line = line.split(" ")
                
                tag = splited_line[0]
                content = " ".join(splited_line[1:])

                if tokens_only:
                    yield tag, gensim.utils.simple_preprocess(content)
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(content), [tag])

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
        sims = self.model.docvecs.most_similar([vector], topn=len(self.model.docvecs))
        return ' '.join(self.train_corpus[sims[0][0]].words)

    def evaluate(self):
        for tag, test_corpus in self.test_corpus:
            print("evaluating", tag, test_corpus)

            inferred_vector = self.model.infer_vector(test_corpus)
            sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))

            print(sims)
            print(inferred_vector)

    def save_model(self, fname):
        save_path = self.data_dir / fname
        self.model.save(str(save_path))

    def load_model(self, fname):
        load_path = self.data_dir / fname
        self.model = gensim.models.doc2vec.Doc2Vec.load(str(load_path))