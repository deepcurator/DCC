import networkx as nx
import glob
from scipy.sparse import block_diag
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import pandas as pd
import scipy
from datetime import datetime

class DocEmbedder:
    def __init__(self, embedder='tfidf', vector_size=16, window=2, workers=4,
                 min_df=1, token_pattern=r'(?u)\b\w\w+\b'):
        self.documents = []
        self.n_docs = 1
        self.vector_size = vector_size
        self.window = window
        self.workers = workers
        self.map = {}
        self.embedder = embedder
        self.min_df = min_df
        self.token_pattern=token_pattern

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
            self.model = TfidfVectorizer(min_df=self.min_df,token_pattern=self.token_pattern)
            self.model.fit_transform(self.documents)
        return self.model

    def embed(self, doc):
        if 'doc2vec' in self.embedder:
            return self.model.infer_vector([doc])
        else:
            result = self.model.transform([doc])
            return result#np.array(result.toarray()[0])


def add_supernode(G):
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        G.add_edge('supernode', nodes[i])



def generate_node_attributes(G, model):
    attribute_vector = []
    # Generate embeddings
    for node in G.nodes():
        attribute_vector.append(model.embed(node))
    return np.array(attribute_vector)


def create_masks(elts, idx_supernodes, tr_fr=0.8, val_fr=1.0):
    # idx supernodes = [78, 218, 289, 536, 613, 773, 861, 1063, 1126, 1229, 1280, 1382, 1460, 1541, 1620, 1802, 2119, 2361, 2398, 2474, 2617, 2698, 2842, 2869, 2956, 3100, 3186, 3280, 3448, 3524, 3602, 3867, 3948, 4024, 4122, 4248, 4369, 4422, 4479, 4593, 4657, 4733, 5064, 5175, 5270, 5510, 5568, 5660, 5789, 5850, 5924, 6045, 6126, 6290, 6985, 7071, 7209, 7444, 7525, 7841, 7987, 8009, 8016]
#    train_ix = idx_supernodes[40]
#    val_ix = idx_supernodes[53]
#    test_ix = idx_supernodes[62]
    
    ### For code (no test?):
    train_ix = idx_supernodes[int(np.ceil(len(idx_supernodes)*tr_fr))-1]
    val_ix = idx_supernodes[int(np.ceil(len(idx_supernodes)*val_fr))-1]

    train = np.zeros((elts,), dtype=bool)
    train[:train_ix-1] = True
    val = np.zeros((elts,), dtype=bool)
    val[train_ix:val_ix-1] = True
    test = np.zeros((elts,), dtype=bool)
    #test[val_ix:] = True
    for idx in idx_supernodes:
        test[idx-1] = True
    return train, val, test

'''
Takes all text and images together with code repos that: 1. have images, 2. <1000 triples
Many hard-coded params!!!
'''
def load_combo(min_df=2):
    print("Load_text_image")
    data_path='../../Data/'
    files = glob.glob('../text2graph/Output/text/*/t2g.triples')
    text_sep=' '
    im_sep='\t'
    code_sep='\t'
    all_triples = []
    all_labels = []    
    papers=[]
    # generate labels/maps for selected papers
    label_file=os.path.join(data_path,'pwc_edited_plt/pwc_edited_plt.csv')
    pwc=pd.read_csv(label_file)
    pwc['tag']=pwc.paper_link.apply(link2fn)
    labels_dict = dict(zip(pwc['tag'], pwc.conference))
    
    # process triples
    for f in files:
        # directory name is the paper tag
        paper_tag=f.split(os.sep)[-2]
        if paper_tag in labels_dict:
            clean_triples=[]
            with open(f) as fd:
                all_lines = fd.readlines()
                clean_triples = [x.strip().replace('https://github.com/deepcurator/DCC/','').split(text_sep) for x in all_lines]
            pwc.loc[pwc.tag==paper_tag,'text2graph']=len(clean_triples)
            # add triples from all image files for this paper
            image_files=glob.glob(data_path+'image/'+paper_tag+'/*.triples')
            img_triples=[]
            for imf in image_files:
                with open(imf) as fd:
                    im_lines=fd.readlines()
                    temp_triples = [x.strip().replace('https://github.com/deepcurator/DCC/','').split(im_sep) for x in im_lines]
                    img_triples.extend(temp_triples)
            pwc.loc[pwc.tag==paper_tag,'img2graph']=len(img_triples)
            clean_triples.extend(img_triples)
            # if has images, include code:
            f=data_path+'pwc_triples/'+paper_tag+'/combined_triples.triples'
            select_rels=['followedBy','calls']
            if len(img_triples)>0 and os.path.exists(f):
                code_triples=[]
                with open(f) as fd:
                    code_lines = fd.readlines()
                    code_triples = [x.strip().replace('https://github.com/deepcurator/DCC/','').split(code_sep) for x in code_lines]
                    if len(select_rels)>0:
                        code_triples =  [t for t in code_triples if t[1] in select_rels]
                pwc.loc[pwc.tag==paper_tag,'code2graph']=len(code_triples)                    
                if len(code_triples)<1000:
                    print('Code {}: {} triples'.format(paper_tag,len(code_triples)))
                    clean_triples.extend(code_triples)
            if len(clean_triples)>0:
                #print('Combined {}'.format(len(clean_triples)))
                all_triples.append(clean_triples)
                all_labels.append(labels_dict[paper_tag])
                papers.append(paper_tag)   
    ### save updated pwc, and save only for those papers added:
    pwc.to_csv('pwc_annotated.csv', header=True, index=False)
    pwc2=pwc[pwc.tag.isin(papers)]
    pwc2.to_csv('pwc_annotated_filtered.csv', header=True, index=False)
    
    return generate_graphs(all_triples, all_labels, min_df)


def load_data(labels_dict, file_expr, sep='\t', min_df=2, select_rels=[], token_pattern='(?u)\b\w\w+\b'):
    files = glob.glob(file_expr)
    print('Select rels: ',select_rels)
    # skip one paper with very large triple file for code
    #files=[x for x in files if '6749-terngrad-ternary-gradients-to-reduce-communication-in-distributed-deep-learning' not in x]
    all_triples = []
    all_labels = []    
    papers = []
    # process triples
    for f in files:
        # directory name is the paper tag
        paper_tag=f.split(os.sep)[-2]
        if paper_tag in labels_dict:
            with open(f) as fd:
                all_lines = fd.readlines()
                clean_triples = [x.strip().replace('https://github.com/deepcurator/DCC/','').split(sep) for x in all_lines]
                if len(select_rels)>0:
                    clean_triples =  [t for t in clean_triples if t[1] in select_rels] 
                if len(clean_triples) > 0 and len(clean_triples)<1000:
                    all_triples.append(clean_triples)
                    all_labels.append(labels_dict[paper_tag])
                    papers.append(paper_tag)
#    all_df = [pd.DataFrame(triples, columns = ['s', 'p','o']) for triples in all_triples]
#    z=pd.concat(all_df,axis=0)
#    ind=z.p.apply(lambda x: 'has_keyword_' not in x and '_size' not in x and 'has_arg' not in x and '#label' not in x).values
#    z2=z[ind]
#    all_triples = [tuple(x) for x in z2.values]
    #temp_df=z2[z2.p.apply(lambda x: '#label' in x)]
    #type_df=z2[z2.p.apply(lambda x: '#type' in x)]    
#    ln=np.array([len(x) for x in all_triples])
#    ind=np.where(ln<10000)[0]
#    print('Skipping {} repos with more than 10K edges'.format(len(ln)-len(ind)))
#    all_triples=[all_triples[i] for i in ind]
#    all_labels=[all_labels[i] for i in ind]
    print("Triple lists: {}; labels: {}".format(len(all_triples),len(all_labels)))
    return generate_graphs(all_triples, all_labels, min_df)

def generate_graphs(all_triples, all_labels, min_df=2):    
    all_graphs = []
    all_As = []
    all_node_attributes = []
    idx_supernodes = []
    idx_labels = []
    g_sizes = []
    total_nodes = 0
    
    ### Build nx graphs
    d2v = DocEmbedder(min_df=min_df)
    for i, repo in enumerate(all_triples):
        G = nx.Graph()
        for triple in repo:
            try:
                s, p, o = triple
                G.add_edge(s,o)
            except:
                pass             
        if G.nodes():
            add_supernode(G)
            all_graphs.append(G)
            all_As.append(nx.adjacency_matrix(G))

            total_nodes = total_nodes + len(G.nodes())
            idx_supernodes.append(total_nodes) # assumes nodes are id
            g_sizes.append(len(G.nodes()))
            idx_labels.append(all_labels[i])

            # Collect node attributes
            [d2v.add_doc(x) for x in G.nodes()]

    d2v.train()    
    # Generate node attributes:  Node attribute NxD feature matrix
    node_attributes=0
    if d2v.embedder=='tfidf':
        for G in all_graphs:
            all_node_attributes.append(scipy.sparse.vstack(generate_node_attributes(G, d2v)))
        node_attributes = scipy.sparse.vstack(all_node_attributes)
    else:
        for G in all_graphs:
            all_node_attributes.append(generate_node_attributes(G, d2v))            
        node_attributes = np.concatenate(all_node_attributes)
    # NxN matrix
    block = block_diag(all_As).tocsr() #block_diag(*[x.todense() for x in all_As])
    
    print('Total nodes = %s' % (total_nodes))
    print('Total edges = %s' % (block.nnz))
    #print('idx supernodes = %s' % (idx_supernodes))
    #print('idx labels = %s' % (idx_labels))

    # NxE binary label matrix where E is the number of classes
    lb = preprocessing.LabelBinarizer()
    idx_binary_labels = lb.fit_transform(idx_labels)
    label_stack = []
    for i, _ in enumerate(idx_supernodes):
        label_stack.append(np.vstack([idx_binary_labels[i]]*g_sizes[i]))
    label_matrix = np.concatenate(label_stack)

    elts, _ = block.shape
    train_mask, val_mask, test_mask = create_masks(elts, idx_supernodes)
    print('Finished Data Loading {}'.format(datetime.now()))
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

def link2fn(x):
    x2=x.split('/')[-1].replace('.html','').replace('.pdf','')
    return(x2)

def load_pwc_labels(myfile):
    pwc=pd.read_csv(myfile)
    fns=pwc.paper_link.apply(link2fn)
    labels_dict = dict(zip(fns, pwc.conference))
    return labels_dict

# maps repo url to conference
def load_repo_labels(myfile):
    pwc=pd.read_csv(myfile)
    labels_dict = dict(zip(pwc.repo_link, pwc.conference))
    return labels_dict

if __name__ == '__main__':
    
    dataset_str='combo'
    data_path = '../../Data/'
    min_df = 2
    if dataset_str == "code":
        sep='\t'
        ### data Arqui used: 67 papers processed by UCI/ 63 have triples
#        out_tag=''
#        label_file='./labels2.csv'
#        file_expr = data_path + 'UCI_TF_Papers/rdf_triples/*/combined_triples.triples' # data Arqui used
#        #file_expr= './old_data/rdf_triples/*/combined_triples.triples'
#        labels_dict = load_labels(label_file)
#        select_rels=[]
        ### PWC data/repos - doesn't fit in memory
        out_tag='_c2g'
        label_file=os.path.join(data_path,'pwc_edited_plt/pwc_edited_plt.csv')
        file_expr=data_path+'pwc_triples/*/combined_triples.triples'
        labels_dict = load_pwc_labels(label_file)
        select_rels=['followedBy','calls']
        ### filter labels to only those that also have images:
        im_files=set()
        files = glob.glob(data_path+'image/*/*.triples')
        for f in files:
            # directory name is the paper tag
            paper_tag=f.split(os.sep)[-2]
            im_files.add(paper_tag)
        print(len(labels_dict))
        papers=list(labels_dict.keys())
        for paper in papers:
            if paper not in im_files:
                del labels_dict[paper]
        print(len(labels_dict))
    elif dataset_str == 'text':
        label_file=os.path.join(data_path,'pwc_edited_plt/pwc_edited_plt.csv')
        out_tag='_t2g'
        file_expr='../text2graph/Output/text/*/t2g.triples'
        sep=' '
        select_rels=[]
        labels_dict = load_pwc_labels(label_file)
    elif dataset_str == 'image':
        label_file=os.path.join(data_path,'pwc_edited_plt/pwc_edited_plt.csv')
        out_tag='_i2g'
        file_expr=data_path+'image/*/*.triples'  # './image/*/*.triples'
        sep='\t'
        select_rels=[]
        labels_dict = load_pwc_labels(label_file)
    elif dataset_str == 'combo':
        label_file=os.path.join(data_path,'pwc_edited_plt/pwc_edited_plt.csv')
        out_tag='_combo'
        sep='\t'
        file_expr=''
        select_rels=[]
        labels_dict = load_pwc_labels(label_file)   

    if dataset_str == 'combo':
        adj, features, labels, train, val, test, idx_supernodes, label_encoder = load_combo()
    else:
        adj, features, labels, train, val, test, idx_supernodes, label_encoder = load_data(labels_dict, file_expr, sep=sep, select_rels=select_rels,  min_df = min_df)


#    np.save('aske'+out_tag+'.graph', adj)
#    np.save('aske'+out_tag+'.allx', features)
#    np.save('aske'+out_tag+'.ally', labels)
#    np.save('aske'+out_tag+'.train', train)
#    np.save('aske'+out_tag+'.val', val)
#    np.save('aske'+out_tag+'.test', test)
