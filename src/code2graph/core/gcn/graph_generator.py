import networkx as nx
import glob
import pdb
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os

class DocEmbedder:
    def __init__(self, embedder='tfidf', vector_size=16, window=2, workers=4):
        self.documents = []
        self.n_docs = 1
        self.vector_size = vector_size
        self.window = window
        self.workers = workers
        self.map = {}
        self.embedder = embedder

    def add_doc(self, doc):
        if 'doc2vec' in self.embedder:
            self.documents.append(TaggedDocument(doc, str(self.n_docs)))
        else:
            self.documents.append(doc)
        self.map[doc] = self.n_docs
        self.n_docs = self.n_docs+1

    def train(self):
        if 'doc2vec' in self.embedder:
            self.model = Doc2Vec(self.documents, vector_size=self.vector_size, window=self.window, min_count=1, workers=self.workers)
        else:
            self.model = TfidfVectorizer()
            self.model.fit_transform(self.documents)
        return self.model

    def embed(self, doc):
        if 'doc2vec' in self.embedder:
            return self.model.infer_vector([doc])
        else:
            result = self.model.transform([doc])
            return np.array(result.toarray()[0])


def add_supernode(G):
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        G.add_edge('supernode', nodes[i])



def generate_node_attributes(G, model):
    attribute_vector = []
    # Generate embeddings
    for i, node in enumerate(G.nodes()):
        attribute_vector.append(model.embed(node))
    return np.array(attribute_vector)


def create_masks(elts, idx_supernodes):
    # idx supernodes = [78, 218, 289, 536, 613, 773, 861, 1063, 1126, 1229, 1280, 1382, 1460, 1541, 1620, 1802, 2119, 2361, 2398, 2474, 2617, 2698, 2842, 2869, 2956, 3100, 3186, 3280, 3448, 3524, 3602, 3867, 3948, 4024, 4122, 4248, 4369, 4422, 4479, 4593, 4657, 4733, 5064, 5175, 5270, 5510, 5568, 5660, 5789, 5850, 5924, 6045, 6126, 6290, 6985, 7071, 7209, 7444, 7525, 7841, 7987, 8009, 8016]

#    train_ix = idx_supernodes[40]
#    val_ix = idx_supernodes[53]
#    test_ix = idx_supernodes[62]
    
    train_ix = idx_supernodes[40]
    val_ix = idx_supernodes[53]
    # test_ix = idx_supernodes[62]

    train = np.zeros((elts,), dtype=bool)
    train[:train_ix-1] = True
    val = np.zeros((elts,), dtype=bool)
    val[train_ix:val_ix-1] = True
    test = np.zeros((elts,), dtype=bool)
    #test[val_ix:] = True
    for idx in idx_supernodes:
        test[idx-1] = True
    return train, val, test


def load_data(labels_dict):
    # files = glob.glob('./text2graph/*/text2graph.triples')
    files = glob.glob('./*/combined_triples.triples')
    all_triples = []
    all_graphs = []
    all_As = []
    all_labels = []
    all_node_attributes = []
    idx_supernodes = []
    idx_labels = []
    g_sizes = []
    total_nodes = 0


    d2v = DocEmbedder()

    # print(labels_dict)
    
    # process triples
    for f in files:
        # print(f)
        with open(f) as fd:
            all_lines = fd.readlines()
            clean_triples = [x.strip() for x in all_lines]
            if f.split(os.sep)[1] in labels_dict:
                all_triples.append(clean_triples)
                # print(f.split(os.sep))
                # all_labels.append(labels_dict[f.split('/')[1]])
                all_labels.append(labels_dict[f.split(os.sep)[1]])

    # Build nx graphs
    for i, repo in enumerate(all_triples):
        G = nx.Graph()
        for triple in repo:
            try:
                s, p, o = triple.split('\t')
            except:
                pass
            G.add_edge(s,o)

 
        if G.nodes():
            add_supernode(G)
            all_graphs.append(G)
            all_As.append(nx.adjacency_matrix(G))

            total_nodes = total_nodes + len(G.nodes())
            idx_supernodes.append(total_nodes)
            g_sizes.append(len(G.nodes()))
            idx_labels.append(all_labels[i])

            # Collect node attributes
            [d2v.add_doc(x) for x in G.nodes()]

    # Generate node attributes
    d2v.train()
    for G in all_graphs:
        all_node_attributes.append(generate_node_attributes(G, d2v))


    print('Total nodes = %s' % (total_nodes))
    print('idx supernodes = %s' % (idx_supernodes))
    print('idx labels = %s' % (idx_labels))

    # NxN matrix
    block = block_diag(*[x.todense() for x in all_As])
    # Node attribute NxD feature matrix
    node_attributes = np.concatenate(all_node_attributes)
    # NxE binary label matrix where E is the number of classes
    lb = preprocessing.LabelBinarizer()
    idx_binary_labels = lb.fit_transform(idx_labels)
    label_stack = []
    for i, _ in enumerate(idx_supernodes):
        label_stack.append(np.vstack([idx_binary_labels[i]]*g_sizes[i]))
    label_matrix = np.concatenate(label_stack)

    elts, _ = block.shape
    train_mask, val_mask, test_mask = create_masks(elts, idx_supernodes)

    return block, node_attributes, label_matrix, train_mask, val_mask, test_mask, idx_supernodes, lb


def show_matrix(mat):
    plt.matshow(mat)
    plt.show()

def load_labels(myfile):
    with open(myfile, mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        labels_dict = {}
        for row in csv_reader:
            labels_dict[row[0]] = row[1]
    return labels_dict

if __name__ == '__main__':
    labels_dict = load_labels('./labels.csv')
    adj, features, labels, train, val, test, idx_supernodes, label_encoder = load_data(labels_dict)
    np.save('aske.graph', adj)
    np.save('aske.allx', features)
    np.save('aske.ally', labels)
    np.save('aske.train', train)
    np.save('aske.val', val)
    np.save('aske.test', test)
