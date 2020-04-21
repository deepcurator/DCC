from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
import tensorflow as tf

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

        return raw_data, raw_tags


class PathContextReader:
    ''' class for preprocessing the data '''
    def __init__(self, path):
        self.bags_train = None
        self.bags_test  = None

        self.path = Path(path).resolve()        

    def read_path_contexts(self):
        self.read_dictionaries()

        self.bags_train = self.read_data(data_path="train.txt")
        self.bags_test  = self.read_data(data_path="test.txt")

        print("Number of unique of words: " + str(len(self.word_count)))
        print("Number of unique of paths: " + str(len(self.path_count)))
        print("Number of unique of targets: " + str(len(self.target_count)))

        print("Number of training samples: " + str(len(self.bags_train)))
        print("Number of testing samples: " + str(len(self.bags_test)))

    def read_dictionaries(self):
        with open(str(self.path / 'reduced_word_count.pkl'), 'rb') as f:
            self.word_count = pickle.load(f)
        with open(str(self.path / 'reduced_word2idx.pkl'), 'rb') as f:
            self.word2idx = pickle.load(f)
        with open(str(self.path / 'reduced_idx2word.pkl'), 'rb') as f:
            self.idx2word = pickle.load(f)

        with open(str(self.path / 'reduced_path_count.pkl'), 'rb') as f:
            self.path_count = pickle.load(f)
        with open(str(self.path / 'reduced_path2idx.pkl'), 'rb') as f:
            self.path2idx = pickle.load(f)
        with open(str(self.path / 'reduced_idx2path.pkl'), 'rb') as f:
            self.idx2path = pickle.load(f)

        with open(str(self.path / 'reduced_target_count.pkl'), 'rb') as f:
            self.target_count = pickle.load(f)
        with open(str(self.path / 'reduced_target2idx.pkl'), 'rb') as f:
            self.target2idx = pickle.load(f)
        with open(str(self.path / 'reduced_idx2target.pkl'), 'rb') as f:
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
                    triple_ids.append([e1,p,e2])

                bags.append((label_ids, triple_ids))

        return bags


class Config:
    def __init__(self):
        # hyperparameters used in tf dataset training. 

        self.epoch = 500
        self.training_batch_size = 256
        self.testing_batch_size = 128

        self.dropout_factor = 0.5
        self.learning_rate = 0.005
        self.embedding_size = 50
        self.code_embedding_size = 50

        self.max_contexts = 200


class Trainer:
    ''' the trainer for code2vec '''
    def __init__(self, path):
        self.config = Config()

        self.reader = PathContextReader(path)
        self.reader.read_path_contexts()
        self.config.path = path

        self.config.num_of_words = len(self.reader.word_count)
        self.config.num_of_paths = len(self.reader.path_count)
        self.config.num_of_tags  = len(self.reader.target_count)

        self.model = code2vec(self.config)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.config.learning_rate)
    
    @tf.function
    def train_step(self, e1, p, e2, tags):
        with tf.GradientTape() as tape:
            code_vectors, attention_weights = self.model.forward(e1, p, e2)
            logits = tf.matmul(code_vectors, self.model.tags_embeddings, transpose_b=True)
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(tags, [-1]), logits=logits))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss
    
    @tf.function
    def test_step(self, e1, p, e2, topk=1):
        code_vectors, _ = self.model.forward(e1, p, e2, train=False)
        logits = tf.matmul(code_vectors, self.model.tags_embeddings, transpose_b=True)
        _, ranks = tf.nn.top_k(logits, k=topk)
        return ranks

    def train_model(self):

        self.train_batch_generator = Generator(self.reader.bags_train, self.config.training_batch_size)
        
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

            if epoch_idx % 5 == 0:
                print("Evaluation Set Test:")
                self.evaluate_model(training_set=False)
                print("Training Set Test:")
                self.evaluate_model(training_set=True)
                self.save_model(epoch_idx)
                self.export_code_embeddings(epoch_idx)

            print('epoch[%d] ---Acc Train Loss: %.5f' % (epoch_idx, acc_loss))

    def evaluate_model(self, training_set = False):
        if training_set: 
            self.test_batch_generator = Generator(self.reader.bags_train, self.config.testing_batch_size)
        else:
            self.test_batch_generator = Generator(self.reader.bags_test, self.config.testing_batch_size)

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        prediction_rank = 0 
        prediction_reciprocal_rank = 0
        nr_predictions = 0

        for batch_idx in range(self.test_batch_generator.number_of_batch):

            data, tag = next(self.test_batch_generator)
            
            e1 = data[:,:,0]
            p  = data[:,:,1]
            e2 = data[:,:,2]
            y  = tag
            
            ranks = self.test_step(e1, p, e2)
            
            ranks_number = tf.where(tf.equal(self.test_step(e1, p, e2, topk=self.config.num_of_tags), tf.cast(tf.expand_dims(y,-1), dtype=tf.int32)))

            for idx, rank_number in enumerate(ranks_number.numpy().tolist()): 
                prediction_rank += (rank_number[1] + 1)
                prediction_reciprocal_rank += 1.0 / (rank_number[1] + 1)

            for idx, rank in enumerate(ranks.numpy().tolist()):
                nr_predictions += 1
                
                original_name = self.reader.idx2target[tag.tolist()[idx]]
                inferred_names = [self.reader.idx2target[target_idx] for target_idx in rank]

                original_subtokens = original_name.split('|')
                

                true_positive = 0
                false_positive = 0
                false_negative = 0

                for inferred_name in inferred_names:
                    inferred_subtokens = inferred_name.split('|')

                    true_positive += sum(1 for subtoken in inferred_subtokens if subtoken in original_subtokens)
                    false_positive += sum(1 for subtoken in inferred_subtokens if subtoken not in original_subtokens)
                    false_negative += sum(1 for subtoken in original_subtokens if subtoken not in inferred_subtokens)

                # if false_positive > 0:
                #     print(original_name)
                #     print(inferred_names)

                true_positives += true_positive
                false_positives += false_positive
                false_negatives += false_negative

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        prediction_rank /= nr_predictions
        prediction_reciprocal_rank /= nr_predictions

        print("\nPrecision: {}, Recall: {}, F1: {} Rank: {} Reciprocal_Rank: {}\n".format(precision, recall, f1, prediction_rank, prediction_reciprocal_rank))
    
    def export_code_embeddings(self, epoch_idx):
        save_path = self.config.path / ('epoch_%d' % epoch_idx)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(str(save_path / "code_labels.tsv"), 'w') as l_export_file:
            for label in self.reader.idx2target.values():
                l_export_file.write(label + "\n")

            parameter = self.model.tags_embeddings
            
            all_ids = list(range(0, int(parameter.shape[0])))
            stored_name = parameter.name.split(':')[0]

            if len(parameter.shape) == 2:
                all_embs = parameter.numpy()
                with open(str(save_path / ("%s.tsv" % stored_name)), 'w') as v_export_file:
                    for idx in all_ids:
                        v_export_file.write("\t".join([str(x) for x in all_embs[idx]]) + "\n")

    def save_model(self, epoch_idx):
        saved_path = self.config.path / ('epoch_%d' % epoch_idx)
        saved_path.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(saved_path / 'model.vec'))

    def load_model(self, path):
        if path.exists():
            self.model.load_weights(str(path / 'model.vec'))

class code2vec(tf.keras.Model): 

    def __init__(self, config):
        super(code2vec, self).__init__()

        self.config = config 

        self.def_parameters()

    def def_parameters(self):        
        emb_initializer = tf.initializers.glorot_normal()
        self.ents_embeddings = tf.Variable(emb_initializer(shape=(self.config.num_of_words, self.config.embedding_size)), name='ents')
        self.path_embeddings = tf.Variable(emb_initializer(shape=(self.config.num_of_paths, self.config.embedding_size)), name='paths')
        self.tags_embeddings = tf.Variable(emb_initializer(shape=(self.config.num_of_tags, self.config.code_embedding_size)), name='tags')
        self.attention_param = tf.Variable(emb_initializer(shape=(self.config.code_embedding_size, 1)), name='attention_param')
        self.transform_matrix= tf.Variable(emb_initializer(shape=(3*self.config.embedding_size, self.config.code_embedding_size)), name='transform')

    def forward(self, e1, p, e2, train=True):
        # e1_e is [batch_size, max_contexts, embeddings size]
        # p_e  is [batch_size, max_contexts, embeddings size]
        # e2_e is [batch_size, max_contexts, embeddings size]
        e1_e = tf.nn.embedding_lookup(params=self.ents_embeddings, ids=e1)
        p_e  = tf.nn.embedding_lookup(params=self.path_embeddings, ids=p)
        e2_e = tf.nn.embedding_lookup(params=self.ents_embeddings, ids=e2)

        # context_emb = [batch_size, max_contexts, 3*embedding_size]        
        context_e = tf.concat([e1_e, p_e, e2_e], axis=-1) 

        # apply a dropout to context emb. 
        if train:
            context_e = tf.nn.dropout(context_e, rate=1-self.config.dropout_factor)

        # flatten context embeddings => [batch_size*max_contexts, 3*embedding_size]
        context_e = tf.reshape(context_e, [-1, 3*self.config.embedding_size])

        # tranform context embeddings -> to [batch_size*max_contexts, code_embedding_size]
        flat_emb = tf.tanh(tf.matmul(context_e, self.transform_matrix))

        # calculate weights => to [batch_size*max_contexts, 1]
        contexts_weights = tf.matmul(flat_emb, self.attention_param)

        # reshapeing context weights => to [batch_size, max_contexts, 1]
        batched_contexts_weights = tf.reshape(contexts_weights, [-1, self.config.max_contexts, 1])

        # calculate softmax for attention weights. 
        attention_weights = tf.nn.softmax(batched_contexts_weights, axis=1)

        # reshaping the embeddings => to [batch_size, max_contexts, code_embedding_size]
        batched_flat_emb = tf.reshape(flat_emb, [-1, self.config.max_contexts, self.config.code_embedding_size])

        # calculating the code vectors => to [batch_size, code_embedding_size]
        code_vectors = tf.reduce_sum(tf.multiply(batched_flat_emb, attention_weights), axis=1)

        return code_vectors, attention_weights