from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
#from gae.input_data import load_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
import graph_generator
import networkx as nx
import scipy

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
#flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
#flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

# passed as variables into code
#flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
#flags.DEFINE_string('dataset', 'code', 'code or text')
#flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')


class RunGAE(object):
    
    def __init__(self, file_expr, label_file, model_str='gcn_ae', file_sep='\t', out_tag='',
                 use_features=True, epochs=100, dropout_rate=0):
        self.model = None
        self.model_str=model_str
        self.out_tag=out_tag
        self.file_expr=file_expr
        self.label_file=label_file
        self.use_features=use_features
        self.dropout_rate=dropout_rate
        self.epochs=epochs
        self.file_sep=file_sep
        
        # Define placeholders
        self.placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }                
        
    def run(self):
        self.labels_dict = graph_generator.load_pwc_labels(self.label_file) #load_labels(self.label_file)   
        n_by_n, x_train, y_train, train_mask, val_mask, test_mask, idx_supernodes, label_encoder = graph_generator.load_data(self.labels_dict, self.file_expr, sep=self.file_sep)
        self.idx_supernodes=idx_supernodes
        adj = nx.adjacency_matrix(nx.from_numpy_array(n_by_n))
        features = scipy.sparse.csr.csr_matrix(x_train)
            
        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()
        self.adj_orig=adj_orig
        
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        adj = adj_train        
        # Some preprocessing
        adj_norm = preprocess_graph(adj)                
        num_nodes = adj.shape[0]


        if not self.use_features:
            features = sp.identity(features.shape[0])  # featureless        
        features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]
        
        # Create model
        if model_str == 'gcn_ae':
            self.model = GCNModelAE(self.placeholders, num_features, features_nonzero)
        elif model_str == 'gcn_vae':
            self.model = GCNModelVAE(self.placeholders, num_features, num_nodes, features_nonzero)
        
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        
        # Optimizer
        with tf.name_scope('optimizer'):
            if model_str == 'gcn_ae':
                opt = OptimizerAE(preds=self.model.reconstructions,
                                  labels=tf.reshape(tf.sparse_tensor_to_dense(self.placeholders['adj_orig'],
                                                                              validate_indices=False), [-1]),
                                  pos_weight=pos_weight,
                                  norm=norm)
            elif model_str == 'gcn_vae':
                opt = OptimizerVAE(preds=self.model.reconstructions,
                                   labels=tf.reshape(tf.sparse_tensor_to_dense(self.placeholders['adj_orig'],
                                                                               validate_indices=False), [-1]),
                                   model=self.model, num_nodes=num_nodes,
                                   pos_weight=pos_weight,
                                   norm=norm)
    
        # Initialize session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        val_roc_score = []
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        
        #import datetime
        #log_dir="logs/gae/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Train model
        for epoch in range(self.epochs): #FLAGS.epochs):        
            t = time.time()
            # Construct feed dictionary
            self.feed_dict = construct_feed_dict(adj_norm, adj_label, features, self.placeholders)
            self.feed_dict.update({self.placeholders['dropout']: self.dropout_rate}) # FLAGS.dropout})
            # Run single weight update
            outs = self.sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=self.feed_dict)
        
            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]
        
            roc_curr, ap_curr = self.get_roc_score(val_edges, val_edges_false)
            val_roc_score.append(roc_curr)
        
        #    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t))
        
        print("Optimization Finished!")
        
        roc_score, ap_score = self.get_roc_score(test_edges, test_edges_false)
        print('Test ROC score: ' + str(roc_score))
        print('Test AP score: ' + str(ap_score))
        
        self.get_embeddings(y_train, label_encoder, out_tag=self.out_tag)

    def get_roc_score(self,edges_pos, edges_neg, emb=None):
        if emb is None:
            self.feed_dict.update({self.placeholders['dropout']: 0})
            emb = self.sess.run(self.model.z_mean, feed_dict=self.feed_dict)
    
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
    
        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(self.adj_orig[e[0], e[1]])
    
        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(self.adj_orig[e[0], e[1]])
    
        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
    
        return roc_score, ap_score
    
    def get_embeddings(self, labels, label_encoder, out_tag=''):
        self.feed_dict.update({self.placeholders['dropout']:0})
        emb = self.sess.run(self.model.z_mean, feed_dict=self.feed_dict)
        labels_txt = label_encoder.inverse_transform(labels)
        
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=300)
        tsne_results = tsne.fit_transform(emb)
        #tsne_results = tsne.fit_transform(np.array(supernodes))
    
        supernodes = []
        supernodes_embeddings = []
        supernodes_labels = []
        colors = []
        for i in self.idx_supernodes:
            supernodes.append(tsne_results[i-1,:])
            supernodes_embeddings.append(emb[i-1,:])
            supernodes_labels.append(labels_txt[i-1])
            color_label = np.where(labels[i-1]==1)
            colors.append(color_label[0][0])
        supernodes = np.array(supernodes)
    
        np.savetxt('uci_embeddings'+out_tag+'.csv', emb, delimiter='\t')
        np.savetxt('uci_supernodes'+out_tag+'.csv', supernodes_embeddings, delimiter='\t')
        np.savetxt('uci_embeddings_labels'+out_tag+'.txt', labels_txt, fmt='%s')
        np.savetxt('uci_supernodes_labels'+out_tag+'.txt', supernodes_labels, fmt='%s')

        import matplotlib.pyplot as plt
        #import seaborn as sns    
        #sns.scatterplot(tsne_results[:,0], tsne_results[:,1], hue=colors)
        #plt.scatter(tsne_results[:,0], tsne_results[:,1], c=colors, cmap='jet') 
        plt.scatter(supernodes[:,0], supernodes[:,1], c=colors, cmap='jet')
        #plt.show()
        plt.savefig('supernode_tsne'+out_tag+'.png')
        plt.clf()

######################################
if __name__ == "__main__":
    model_str = "gcn_ae" #FLAGS.model
    dataset_str = "code" #"text"  #FLAGS.dataset

    #label_file='./labels.csv'    
    label_file='../../../pwc_edited_plt/pwc_edited_plt.csv'
    doCode=(dataset_str=='code')
    if doCode:
        out_tag=''
        file_expr='./rdf_triples/*/combined_triples.triples'
        sep='\t'
    else:
        out_tag='_t2g'
        #file_expr='./text2graph/*/text2graph.triples'
        #file_expr='../text2graph/Output/text/*.txt'
        file_expr='../text2graph/Output/text/*/t2g.triples'
        sep=' '
    runner=RunGAE(file_expr, label_file, model_str=model_str, file_sep=sep, out_tag=out_tag)
    runner.run()