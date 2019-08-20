import tensorflow as tf
from pathlib import Path
from glob import glob
import code, os, timeit
import numpy as np

class PathContextReader:
    ''' class for preprocessing the data '''
    def __init__(self, config):
        self.config = config
        self.path = Path('C:\\Users\\louisccc\\Documents\\GitHub\\DCC\\src\\code2graph\\test\\triples_ast_function').resolve()

        self.function_paths = glob("%s/**/*.txt" % str(self.path), recursive=True)


        self.bags = []

    def read_path_contexts(self):
        entities_set = set()
        paths_set    = set()
        tags_set     = set()

        # 1st iteration to build dictionaries
        for function_path in self.function_paths:
            
            filename = os.path.splitext(Path(function_path).name)[0]
            for tag in filename.split('.'):
                tags_set.add(tag)

            with open(function_path, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    elements = line.split('\t')
                    head, path, tail = elements[0].strip(), elements[1].strip(), elements[2].strip()
                    entities_set.add(head)
                    entities_set.add(tail)
                    paths_set.add(path)
                    
        self.entities = np.sort(list(entities_set))
        self.paths    = np.sort(list(paths_set))
        self.tags     = np.sort(list(tags_set))

        self.entity2idx = {v: k for k, v in enumerate(self.entities)}
        self.idx2entity = {v: k for k, v in self.entity2idx.items()}
        self.path2idx   = {v: k for k, v in enumerate(self.paths)}
        self.idx2path   = {v: k for k, v in self.path2idx.items()}
        self.tag2idx    = {v: k for k, v in enumerate(self.tags)}
        self.idx2tag    = {v: k for k, v in self.tag2idx.items()}

        # 2nd iteration to constracut bags of contextpaths in indices. 
        for function_path in self.function_paths:
            function_bag = {} 
            function_bag['tags'] = []
            function_bag['path_contexts'] = []

            filename = os.path.splitext(Path(function_path).name)[0]
            print(filename.split('.'))
            for tag in filename.split('.'):
                function_bag['tags'].append(self.tag2idx[tag])
            
            with open(function_path, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    elements = line.split('\t')
                    head, path, tail = elements[0].strip(), elements[1].strip(), elements[2].strip()
                    head_idx, path_idx, tail_idx = self.entity2idx[head], self.path2idx[path], self.entity2idx[tail]
                    function_bag['path_contexts'].append((head_idx, path_idx, tail_idx))

            self.bags.append(function_bag)


class Config:
    def __init__(self, args):
        self.path = args.function_path 


class Trainer:
    ''' the trainer for code2vec '''
    def __init__(self):
        self.reader = PathContextReader(None)
        self.reader.read_path_contexts()

        self.model = code2vec(None)

    def build_model(self):

        self.model.def_inputs()
        self.model.def_parameters()
        self.model.def_loss()

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer()
        grads = optimizer.compute_gradients(self.model.loss)
        self.op_train = optimizer.apply_gradients(grads, global_step=self.global_step)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train_model(self):

        for epoch_idx in range(0, 500):
            
            acc_loss = 0

            num_batch = 10 # get the number of batches

            start_time = timeit.default_timer()

            for batch_idx in range(num_batch):
                # data = self.reader.getinputs(batch_idx)
                # h = data[0]
                # p = data[1]
                # t = data[2]
                # tags= data[3]
                
                # feed_dict = {
                #     self.model.h: h,
                #     self.model.p: p,
                #     self.model.t: t,
                #     self.model.tags: tags
                # }
                
                # _, step, loss = self.sess.run([self.op_train, self.global_step, self.model.loss], feed_dict)

                # acc_loss += loss
                pass
                # print('[%.2f sec](%d/%d): -- loss: %.5f' % (timeit.default_timer() - start_time, batch_idx, num_batch, loss), end='\r')

            # print('iter[%d] ---Train Loss: %.5f ---time: %.2f' % (epoch_idx, acc_loss, timeit.default_timer() - start_time))

class code2vec: 

    def __init__(self, config):
        self.embedding_size = 128
        self.code_embedding_size = 50

        self.tot_tags = 0
        self.tot_ents = 0
        self.tot_paths = 0

        self.dropout_factor = 0.75
        
        self.max_contexts = 300


    def def_inputs(self):
        # training placeholders.
        # the training input will be mini-batch function context paths.
        self.h = tf.placeholder(tf.int32, [None, self.max_contexts])
        self.p = tf.placeholder(tf.int32, [None, self.max_contexts])
        self.t = tf.placeholder(tf.int32, [None, self.max_contexts])
        self.tags = tf.placeholder(tf.int32, [None])

        # testing placeholders
        # self.test_h_batch = tf.placeholder(tf.int32, [None])
        # self.test_p_batch = tf.placeholder(tf.int32, [None])
        # self.test_t_batch = tf.placeholder(tf.int32, [None])

    def def_parameters(self):
        
        self.ents_embeddings = tf.get_variable(name='ents', shape=[self.tot_ents + 1, self.embedding_size], dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        self.path_embeddings = tf.get_variable('paths', shape=[self.tot_paths + 1, self.embedding_size], dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        self.tags_embeddings = tf.get_variable('tags', shape=[self.tot_tags + 1, self.code_embedding_size], dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        self.attention_param = tf.get_variable('attention_param', shape=[self.code_embedding_size, 1], dtype=tf.float32, 
                                          initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        self.transform_matrix= tf.get_variable('transform', shape=[3*self.embedding_size, self.code_embedding_size], dtype=tf.float32, 
                                          initializer=tf.contrib.layers.xavier_initializer(uniform=True))

    def def_loss(self):
        # head_ent_embs is [batch_size, max_contexts, embeddings size]
        # path_emb is [batch_size, max_contexts, embeddings size]
        # tail_ent_emb is [batch_size, max_contexts, embeddings size]
        head_ent_emb = tf.nn.embedding_lookup(params=self.ents_embeddings, ids=self.h)
        path_emb     = tf.nn.embedding_lookup(params=self.path_embeddings, ids=self.p)
        tail_ent_emb = tf.nn.embedding_lookup(params=self.ents_embeddings, ids=self.t)

        # tags_emb     = tf.nn.embedding_lookup(params=self.tags_embeddings, id=self.tags)

        # context_emb = [batch_size, max_contexts, 3*embedding_size]        
        context_emb = tf.concat([head_ent_emb, path_emb, tail_ent_emb], axis=-1) 

        # apply a dropout to context emb. 
        context_emb = tf.nn.dropout(context_emb, self.dropout_factor)

        # flatten context embeddings => [batch_size*max_contexts, 3*embedding_size]
        context_emb = tf.reshape(context_emb, [-1, 3*self.embedding_size])

        # tranform context embeddings -> to [batch_size*max_contexts, code_embedding_size]
        flat_emb = tf.tanh(tf.matmul(context_emb, self.transform_matrix))

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

        # calculate the loss [batch_size, code embedding] * [code_embedding, tot_tags]
        logits = tf.matmul(code_vectors, self.tags_embeddings, transpose_b=True)

        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.tags, [-1]),
                logits=logits))


        # 
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


def main():

    trainer = Trainer()
    trainer.build_model()
    trainer.train_model()

    code.interact(local=locals())
    
if __name__ == "__main__":
    main()
