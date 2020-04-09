import tensorflow as tf
from pathlib import Path
from glob import glob
import code, os, timeit, pickle
import numpy as np
from tqdm import tqdm

class Generator:

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        
        self.number_of_batch = len(data) // self.batch_size
        self.random_ids = np.random.permutation(len(data))
        
        self.batch_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        data = self.data
        pos_start = self.batch_size * self.batch_idx
        pos_end   = self.batch_size * (self.batch_idx+1)

        raw_data = np.asarray([data[x][1] for x in self.random_ids[pos_start:pos_end]])
        raw_tags = np.asarray([int(data[x][0]) for x in self.random_ids[pos_start:pos_end]])

        self.batch_idx += 1
        if  self.batch_idx == self.number_of_batch:
            self.batch_idx = 0
        # [t1, t2, t3], [t1, t2, t3]
        # [3],          [4]
        return raw_data, raw_tags


class PathContextReader:
    ''' class for preprocessing the data '''
    def __init__(self, config, path):
        self.config = config
        
        self.path = Path(path).resolve()
        
        self.bags_train = []
        self.bags_test  = []

    def read_path_contexts(self):
        self.read_dictionaries()
        self.bags_train = self.read_data(data_path="train.txt")
        self.bags_test  = self.read_data(data_path="test.txt")

    def read_dictionaries(self):
        with open(str(self.path / 'word_count.pkl'), 'rb') as f:
            self.word_count = pickle.load(f)
        with open(str(self.path / 'word2idx.pkl'), 'rb') as f:
            self.word2idx = pickle.load(f)
        with open(str(self.path / 'idx2word.pkl'), 'rb') as f:
            self.idx2word = pickle.load(f)

        with open(str(self.path / 'path_count.pkl'), 'rb') as f:
            self.path_count = pickle.load(f)
        with open(str(self.path / 'path2idx.pkl'), 'rb') as f:
            self.path2idx = pickle.load(f)
        with open(str(self.path / 'idx2path.pkl'), 'rb') as f:
            self.idx2path = pickle.load(f)

        with open(str(self.path / 'target_count.pkl'), 'rb') as f:
            self.target_count = pickle.load(f)
        with open(str(self.path / 'target2idx.pkl'), 'rb') as f:
            self.target2idx = pickle.load(f)
        with open(str(self.path / 'idx2target.pkl'), 'rb') as f:
            self.idx2target = pickle.load(f)

    def read_data(self, data_path="train.txt"):
        bags=[]

        with open((self.path / data_path), 'r') as file:
            for function_line in file:
                splited_function_line = function_line.split(" ")
                label_ids = splited_function_line[0]
                triples = splited_function_line[1:]
                triple_ids = []

                for triple in triples:
                    splited_triple = triple.split('\t')
                    
                    if len(splited_triple) != 3: 
                        assert False, "Weird non-triple data row."

                    e1, p, e2 = int(splited_triple[0]), int(splited_triple[1]), int(splited_triple[2])
                    triple_ids.append((e1,p,e2))

                bags.append((label_ids, triple_ids))

        # bags = [('label_id', [triples])]
        return bags


class Config:
    def __init__(self):
        pass


class Trainer:
    ''' the trainer for code2vec '''
    def __init__(self, path):
        self.reader = PathContextReader(None, path)
        self.reader.read_path_contexts()

        self.config = Config()
        
        self.config.num_of_words = len(self.reader.word_count)
        self.config.num_of_paths = len(self.reader.path_count)
        self.config.num_of_tags  = len(self.reader.target_count)
        self.config.epoch = 1

        self.model = code2vec(self.config)
        self.model.def_parameters()
        self.optimizer = tf.keras.optimizers.Adam()
        
    def train_model(self):

        self.train_batch_generator = Generator(self.reader.bags_train, 2)
        for epoch_idx in tqdm(range(self.config.epoch)):
            
            acc_loss = 0

            for batch_idx in range(self.train_batch_generator.number_of_batch):
                data, tag = next(self.train_batch_generator)

                e1 = data[:,:,0]
                p  = data[:,:,1]
                e2 = data[:,:,2]
                y  = tag

                loss = self.train_step(e1, p, e2, y)

                acc_loss += loss

            print('epoch[%d] ---Acc Train Loss: %.5f' % (epoch_idx, acc_loss))

    @tf.function
    def train_step(self, e1, p, e2, tags):
        with tf.GradientTape() as tape:
            code_vectors, attention_weights = self.model.forward(e1, p, e2)
            logits = tf.matmul(code_vectors, self.model.tags_embeddings, transpose_b=True)
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(tags, [-1]), logits=logits))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss
    
    # @tf.function
    def test_step(self, e1, p, e2, topk=10):
        code_vectors, _ = self.model.forward(e1, p, e2)
        logits = tf.matmul(code_vectors, self.model.tags_embeddings, transpose_b=True)
        _, ranks = tf.nn.top_k(logits, k=topk)
        return ranks

    def evaluate_model(self):
        self.test_batch_generator = Generator(self.reader.bags_test, 2)
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        nr_predictions = 0

        for batch_idx in range(self.test_batch_generator.number_of_batch):
            data, tag = next(self.test_batch_generator)
            e1 = data[:,:,0]
            p  = data[:,:,1]
            e2 = data[:,:,2]
            y  = tag

            ranks = self.test_step(e1, p, e2)
            
            for idx,rank in enumerate(ranks.numpy().tolist()):
                nr_predictions += 1
                original_name = self.reader.idx2target[tag.tolist()[idx]]
                inferred_names = [self.reader.idx2target[target_idx] for target_idx in rank]
                
                original_subtokens = original_name.split('|')

                for inferred_name in inferred_names:
                    inferred_subtokens = inferred_name.split('|')
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

class code2vec(tf.keras.Model): 

    def __init__(self, config):
        super(code2vec, self).__init__()

        self.embedding_size = 128
        self.code_embedding_size = 50

        self.tot_tags = config.num_of_tags
        self.tot_ents =  config.num_of_words
        self.tot_paths = config.num_of_paths 

        self.dropout_factor = 0.75
        
        self.max_contexts = 200

        self.def_parameters()

    def def_parameters(self):        
        emb_initializer = tf.initializers.glorot_normal()
        self.ents_embeddings = tf.Variable(emb_initializer(shape=(self.tot_ents, self.embedding_size)), name='ents')
        self.path_embeddings = tf.Variable(emb_initializer(shape=(self.tot_paths, self.embedding_size)), name='paths')
        self.tags_embeddings = tf.Variable(emb_initializer(shape=(self.tot_tags, self.code_embedding_size)), name='tags')
        self.attention_param = tf.Variable(emb_initializer(shape=(self.code_embedding_size, 1)), name='attention_param')
        self.transform_matrix= tf.Variable(emb_initializer(shape=(3*self.embedding_size, self.code_embedding_size)), name='transform')

    def forward(self, e1, p, e2):
        # e1_e is [batch_size, max_contexts, embeddings size]
        # p_e  is [batch_size, max_contexts, embeddings size]
        # e2_e is [batch_size, max_contexts, embeddings size]
        
        e1_e = tf.nn.embedding_lookup(params=self.ents_embeddings, ids=e1)
        p_e  = tf.nn.embedding_lookup(params=self.path_embeddings, ids=p)
        e2_e = tf.nn.embedding_lookup(params=self.ents_embeddings, ids=e2)

        # context_emb = [batch_size, max_contexts, 3*embedding_size]        
        context_e = tf.concat([e1_e, p_e, e2_e], axis=-1) 

        # apply a dropout to context emb. 
        context_e = tf.nn.dropout(context_e, rate=1-self.dropout_factor)

        # flatten context embeddings => [batch_size*max_contexts, 3*embedding_size]
        context_e = tf.reshape(context_e, [-1, 3*self.embedding_size])

        # tranform context embeddings -> to [batch_size*max_contexts, code_embedding_size]
        flat_emb = tf.tanh(tf.matmul(context_e, self.transform_matrix))

        # calculate weights => to [batch_size*max_contexts, 1]
        contexts_weights = tf.matmul(flat_emb, self.attention_param)

        # reshapeing context weights => to [batch_size, max_contexts, 1]
        batched_contexts_weights = tf.reshape(contexts_weights, [-1, self.max_contexts, 1])

        # calculate softmax for attention weights. 
        attention_weights = tf.nn.softmax(batched_contexts_weights, axis=1)

        # reshaping the embeddings => to [batch_size, max_contexts, code_embedding_size]
        batched_flat_emb = tf.reshape(flat_emb, [-1, self.max_contexts, self.code_embedding_size])

        # calculating the code vectors => to [batch_size, code_embedding_size]
        code_vectors = tf.reduce_sum(tf.multiply(batched_flat_emb, attention_weights), axis=1)

        return code_vectors, attention_weights

    def evaluation(self):
        pass


def main():

    trainer = Trainer()
    trainer.build_model()
    trainer.train_model()

    code.interact(local=locals())
    
if __name__ == "__main__":
    main()
