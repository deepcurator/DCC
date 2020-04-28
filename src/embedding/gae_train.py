from __future__ import division
from __future__ import print_function

import time
import os
import glob
import sys

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
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple #, mask_test_edges
import graph_generator
import networkx as nx
import scipy
from datetime import datetime

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

def generate_random_pairs(N,size):
    e1=np.random.randint(low=0,high=N, size=size)
    e2=np.random.randint(low=0,high=N, size=size)
    return set(x for x in zip(e1,e2))

'''
Reimplemented for efficiency (compared to gae package, preprocessing.py)
Here we use sets of triples ie sparse rep, which is faster than original
'''
def mask_test_edges2(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Efficiently check that diag is zero: DmitriyFradkin
    assert np.sum(adj.diagonal()) == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    data = np.ones(train_edges.shape[0])
    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    #def ismember(a, b, tol=5):
    #    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    #    return np.any(rows_close)

    print('Generating test_edges_false {}'.format(datetime.now()))
    ### all edges - symmetric 
    edges_all_set = set([(x[0],x[1]) for x in edges_all])
    # generate initial set randomly:
    test_edges_false = generate_random_pairs(adj.shape[0], len(test_edges))
    # make sure it doesn't have real edges:
    test_edges_false = test_edges_false - edges_all_set
    # add as many edges as needed:
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j or (idx_i, idx_j) in edges_all_set:
            continue
        if (idx_j, idx_i) in test_edges_false or (idx_i, idx_j) in test_edges_false:
                continue
        test_edges_false.add((idx_i, idx_j))

    print('Generating val_edges_false {}'.format(datetime.now()))
    val_edges_false = generate_random_pairs(adj.shape[0], len(val_edges))
    # remove edges already existing or in test_false:
    val_edges_false = val_edges_false - edges_all_set
    val_edges_false = val_edges_false - test_edges_false    
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j or (idx_i, idx_j) in edges_all_set:
            continue
        if (idx_i, idx_j) in test_edges_false or (idx_j, idx_i) in test_edges_false:
            continue
        if (idx_i, idx_j) in val_edges_false or (idx_j, idx_i) in val_edges_false:
            continue
        val_edges_false.add((idx_i, idx_j))

#    assert ~ismember(test_edges_false, edges_all)
#    assert ~ismember(val_edges_false, edges_all)
#    assert ~ismember(val_edges, train_edges)
#    assert ~ismember(test_edges, train_edges)
#    assert ~ismember(val_edges, test_edges)

    # convert sets to numpy arrays:
    test_edges_false=np.array([np.array(x) for x in test_edges_false])
    val_edges_false=np.array([np.array(x) for x in val_edges_false])
    
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


class RunGAE(object):
    
    def __init__(self, file_expr, labels_dict, model_str='gcn_ae', file_sep='\t', out_dir='', out_tag='',
                 min_valid_triples=0, use_features=True, select_rels=[],  epochs=100, dropout_rate=0):
        self.model = None
        self.model_str=model_str
        self.out_tag=out_tag
        self.file_expr=file_expr
        self.labels_dict=labels_dict
        self.min_valid_triples=min_valid_triples
        self.use_features=use_features
        self.dropout_rate=dropout_rate
        self.epochs=epochs
        self.file_sep=file_sep
        self.select_rels=select_rels
        self.out_dir=out_dir
        
        # Define placeholders
        self.placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }                
        
    def run(self):
        if self.file_expr == '':
            # text-image-code combination
            n_by_n, x_train, y_train, train_mask, val_mask, test_mask, idx_supernodes, label_encoder = graph_generator.load_combo(self.labels_dict)
        else:
            n_by_n, x_train, y_train, train_mask, val_mask, test_mask, idx_supernodes, label_encoder = graph_generator.load_data(self.labels_dict, self.file_expr,min_valid_triples=self.min_valid_triples,sep=self.file_sep, select_rels=self.select_rels)
        self.idx_supernodes=idx_supernodes
        adj = nx.adjacency_matrix(nx.from_scipy_sparse_matrix(n_by_n)) #nx.adjacency_matrix(nx.from_numpy_array(n_by_n))
        features = scipy.sparse.csr.csr_matrix(x_train)
            
        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()
        self.adj_orig=adj_orig
        
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges2(adj)
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
        
        [supernodes, supernodes_embeddings, supernodes_labels]=self.get_embeddings(y_train, label_encoder)
        self.supernodes=[supernodes, supernodes_embeddings, supernodes_labels]

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
    
    def get_embeddings(self, labels, label_encoder):
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
    
        np.savetxt(self.out_dir+'dcc_embeddings'+self.out_tag+'.csv', emb, delimiter='\t')
        np.savetxt(self.out_dir+'dcc_supernodes'+self.out_tag+'.csv', supernodes_embeddings, delimiter='\t')
        np.savetxt(self.out_dir+'dcc_embeddings_labels'+self.out_tag+'.txt', labels_txt, fmt='%s')
        np.savetxt(self.out_dir+'dcc_supernodes_labels'+self.out_tag+'.txt', supernodes_labels, fmt='%s')
        return(supernodes, supernodes_embeddings, supernodes_labels)
        
######################################
if __name__ == "__main__":
    dataset_str=sys.argv[1]
    #dataset_str = "code" #image" #"text" "code" ###FLAGS.dataset
    model_str = "gcn_ae" #FLAGS.model
    data_path = '../../Data/'

    label_file=os.path.join(data_path,'pwc_edited_plt/pwc_edited_plt.csv')
    labels_dict = load_pwc_labels(label_file)
    select_rels=[]
    sep='\t'
    min_df = 2

    if dataset_str == "code":
        ### data initially used: 67 papers processed by UCI/ 63 have triples
#        out_tag=''
#        label_file='./labels2.csv'
#        file_expr = data_path + 'UCI_TF_Papers/rdf_triples/*/combined_triples.triples' # data generated by UCI - small enough, used for embedding
#        labels_dict = graph_generator.load_labels(label_file)
#        select_rels=[]
        ### PWC data/repos - doesn't fit in memory
        out_tag='_c2g'
        min_valid_triples=3        
        file_expr=data_path+'pwc_triples/*/combined_triples.triples'
        select_rels=['followedBy','calls']
        ### filter labels to only those that also have images:
        im_files=set()
        files = glob.glob(data_path+'image/*/*.triples')
        for f in files:
            # directory name is the paper tag
            paper_tag=f.split(os.sep)[-2]
            im_files.add(paper_tag)
        papers=list(labels_dict.keys())
        for paper in papers:
            if paper not in im_files:
                del labels_dict[paper]
    elif dataset_str == 'text':
        label_file=os.path.join(data_path,'pwc_edited_plt/pwc_edited_plt.csv')
        out_tag='_t2g'
        file_expr='../text2graph/Output/text/*/t2g.triples'
        sep=' '
        min_valid_triples=4
    elif dataset_str == 'image':
        out_tag='_i2g'
        file_expr=data_path+'image/*/*.triples'  # './image/*/*.triples'
        min_valid_triples=0        
    else: # dataset_str == 'combo'
        out_tag='_'+dataset_str
        file_expr=''
        min_valid_triples=0 
    print(dataset_str+" : "+out_tag)
    runner=RunGAE(file_expr, labels_dict, model_str=model_str, file_sep=sep, out_dir='results/',out_tag=out_tag, min_valid_triples=min_valid_triples, select_rels=select_rels)
    runner.run()