import tensorflow as tf
from pathlib import Path
from glob import glob
import code
import numpy as np
class Config:
    def __init__(self, args):
        self.path = args.function_path 


class PathContextReader:
    ''' class for preprocessing the data '''
    def __init__(self, config):
        self.config = config
        self.path = Path('C:\\Users\\AICPS\\Documents\\GitHub\\louisccc-DCC\\src\\code2graph\\test\\triples_ast_function').resolve()

        self.function_paths = glob("%s/**/*.txt" % str(self.path), recursive=True)

        self.bags = []

        self.entities = []
        self.paths = []
        
        self.entities_set = set()
        self.paths_set = set()

    def read_path_contexts(self):
        entities_set = set()
        paths_set    = set()

        for function_path in self.function_paths:
            with open(function_path, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    elements = line.split('\t')
                    head, path, tail = elements[0].strip(), elements[1].strip(), elements[2].strip()
                    entities_set.add(head)
                    entities_set.add(tail)
                    paths_set.add(path)
                    
        self.entities = np.sort(list(entities_set))
        self.paths    = np.sort(list(paths_set))

        self.entity2idx = {v: k for k, v in enumerate(self.entities)}
        self.idx2entity = {v: k for k, v in self.entity2idx.items()}
        self.path2idx   = {v: k for k, v in enumerate(self.paths_set)}
        self.idx2path   = {v: k for k, v in self.path2idx.items()}

class Trainer:
    ''' the trainer for code2vec '''
    def __init__(self):
        pass

    def build_model(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train_model(self):
        pass

preparer = PathContextReader(None)
code.interact(local=locals())
preparer.read_path_contexts()

class code2vec: 

    def __init__(self, config):
        self.embedding_size = 128
        self.tot_tags = 0
        self.tot_ents = 0
        self.tot_paths = 0

        self.dropout_factor = 0.75

    def def_inputs(self):
        self.h = tf.placeholder(tf.int32, [None])
        self.p = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        
        self.test_h_batch = tf.placeholder(tf.int32, [None])
        self.test_p_batch = tf.placeholder(tf.int32, [None])
        self.test_t_batch = tf.placeholder(tf.int32, [None])

    def def_parameters(self):
        
        self.tags_embeddings = tf.get_variable('tags', shape=(self.tot_tags + 1, self.embedding_size), dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))

        self.ents_embeddings = tf.get_variable('ents', shape=(self.tot_ents + 1, self.embedding_size), dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))

        self.path_embeddings = tf.get_variable('paths', shape=(self.tot_paths + 1, self.embedding_size), dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))


    def def_loss(self):
        pass
        # max_contexts = self.config.MAX_CONTEXTS

        # source_word_embed = tf.nn.embedding_lookup(params=words_vocab, ids=source_input)  # (batch, max_contexts, dim)
        # path_embed = tf.nn.embedding_lookup(params=paths_vocab, ids=path_input)  # (batch, max_contexts, dim)
        # target_word_embed = tf.nn.embedding_lookup(params=words_vocab, ids=target_input)  # (batch, max_contexts, dim)

        # context_embed = tf.concat([source_word_embed, path_embed, target_word_embed],
        #                           axis=-1)  # (batch, max_contexts, dim * 3)
        # if not is_evaluating:
        #     context_embed = tf.nn.dropout(context_embed, keep_prob1)

        # flat_embed = tf.reshape(context_embed, [-1, self.config.EMBEDDINGS_SIZE * 3])  # (batch * max_contexts, dim * 3)
        # transform_param = tf.get_variable('TRANSFORM',
        #                                   shape=(self.config.EMBEDDINGS_SIZE * 3, self.config.EMBEDDINGS_SIZE * 3),
        #                                   dtype=tf.float32)

        # flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))  # (batch * max_contexts, dim * 3)

        # contexts_weights = tf.matmul(flat_embed, attention_param)  # (batch * max_contexts, 1)
        # batched_contexts_weights = tf.reshape(contexts_weights,
        #                                       [-1, max_contexts, 1])  # (batch, max_contexts, 1)
        # mask = tf.log(valid_mask)  # (batch, max_contexts)
        # mask = tf.expand_dims(mask, axis=2)  # (batch, max_contexts, 1)
        # batched_contexts_weights += mask  # (batch, max_contexts, 1)
        # attention_weights = tf.nn.softmax(batched_contexts_weights, axis=1)  # (batch, max_contexts, 1)

        # batched_embed = tf.reshape(flat_embed, shape=[-1, max_contexts, self.config.EMBEDDINGS_SIZE * 3])
        # code_vectors = tf.reduce_sum(tf.multiply(batched_embed, attention_weights),
        #                                           axis=1)  # (batch, dim * 3)

        # return code_vectors, attention_weights

    def evaluation(self):
        pass