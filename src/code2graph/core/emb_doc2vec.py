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
        self.test_corpus = list(self.read_dataset(self.test_path, tokens_only=True))

        self.train_tags = list(self.read_keywords(self.train_path))

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

    def read_keywords(self, fname):
        with open(fname, encoding="iso-8859-1") as f:
            for line in f:
                splited_line = line.split(" ")
                
                tag = splited_line[0]
                yield tag
                
    def train_model(self, vector_size=50, window=2, min_count=2, epochs=100, workers=4):
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, window=window, 
                                                   min_count=min_count, epochs=epochs,
                                                   workers=workers)
        self.model.build_vocab(self.train_corpus)
        # import pdb; pdb.set_trace()
        self.model.train(self.train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def dump_vectors(self):
        file_path = self.data_dir / "doc2vec_vectors.txt"
        self.model.docvecs.save_word2vec_format(str(file_path))

    def get_subtokens(self, name):
        function_names = name.split('.')[-1].split('|')
        return function_names

    def evaluate(self):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        nr_predictions = 0

        for tag, keywords_doc in self.test_corpus:
            nr_predictions += 1

            inferred_vector = self.model.infer_vector(keywords_doc)
            sims = self.model.docvecs.most_similar([inferred_vector], topn=10)

            inferred_names = [sim[0] for sim in sims if sim[0] in self.train_tags][:1]
            print(tag, inferred_names)
            original_subtokens = self.get_subtokens(tag)
            
            for inferred_name in inferred_names:
                inferred_subtokens = self.get_subtokens(inferred_name)
                print(original_subtokens, inferred_subtokens)
                true_positives += sum(1 for subtoken in inferred_subtokens if subtoken in original_subtokens)
                false_positives += sum(1 for subtoken in inferred_subtokens if subtoken not in original_subtokens)
                false_negatives += sum(1 for subtoken in original_subtokens if subtoken not in inferred_subtokens)

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
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
