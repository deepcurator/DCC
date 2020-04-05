import gensim
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

    def train_model(self, vector_size=50, window=2, min_count=2, epochs=100, workers=4):
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

    def get_subtokens(self, name):
        hierarchy = name.split('.')[:-1]
        function_name = name.split('.')[-1].split('|')
        return hierarchy + function_name

    def evaluate(self):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        nr_predictions = 0

        for tag, test_corpus in self.test_corpus:
            print("evaluating", tag, test_corpus)
            nr_predictions += 1

            inferred_vector = self.model.infer_vector(test_corpus)
            sims = self.model.docvecs.most_similar([inferred_vector], topn=10)

            inferred_names = [sim[0] for sim in sims]

            original_subtokens = self.get_subtokens(tag)
            
            for inferred_name in inferred_names:
                inferred_subtokens = self.get_subtokens(inferred_name)
                true_positives += sum(1 for subtoken in inferred_subtokens if subtoken in original_subtokens)
                false_positives += sum(1 for subtoken in inferred_subtokens if subtoken not in original_subtokens)
                false_negatives += sum(1 for subtoken in original_subtokens if subtoken not in inferred_subtokens)

        true_positives /= nr_predictions
        false_positives /= nr_predictions
        false_negatives /= nr_predictions
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        print("Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))

    def save_model(self, fname):
        save_path = self.data_dir / fname
        self.model.save(str(save_path))

    def load_model(self, fname):
        load_path = self.data_dir / fname
        self.model = gensim.models.doc2vec.Doc2Vec.load(str(load_path))