import random
import os, sys
import collections

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn import gaussian_process

def rand(size, a, b, decimals=4):
    res = np.random.random_sample(size)*(b-a)+a
    if decimals is not None:
        return np.around(res, decimals=decimals)
    return res

def split_network(G, N):
    sc = SpectralClustering(N, affinity='precomputed')
    return sc.fit_predict(nx.adjacency_matrix(G))

def random_walk_induced_graph_sampling(G, N, T=100, growth_size=2):
    # Refer to https://github.com/Ashish7129/Graph-Sampling
    G = nx.convert_node_labels_to_integers(G, 0, 'default', True)
    for n, data in G.nodes(data=True):
        G.node[n]['id'] = n
    n_node = G.number_of_nodes()
    temp_node = random.randint(0, n_node-1)
    sampled_nodes = set([G.node[temp_node]['id']])
    iter_ = 1
    nodes_before_t_iter = 0
    curr_node = temp_node
    while len(sampled_nodes) != N:
        edges = [n for n in G.neighbors(curr_node)]
        index_of_edge = random.randint(0, len(edges)-1)
        chosen_node = edges[index_of_edge]
        sampled_nodes.add(G.node[chosen_node]['id'])
        curr_node = chosen_node
        iter_ += 1
        if iter_ % T == 0:
            if (len(sampled_nodes)-nodes_before_t_iter < growth_size):
                curr_node = random.randint(0, n_node-1)
            nodes_before_t_iter = len(sampled_nodes)
    sampled_graph = G.subgraph(sampled_nodes)
    return sampled_graph

def write_with_create(path):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return open(path, 'w')

def load_graph(edgelist_filename, label_name=None):
    G = nx.read_edgelist(edgelist_filename, nodetype=int)
    if label_name is not None:
        labels = np.loadtxt(label_name, dtype=int)
        ### multi-label
        l = collections.defaultdict(list)
        for i, j in labels:
            l[i].append(j)
        ### Warning:: The call order of arguments `values` and `name` switched between v1.x & v2.x.
        nx.set_node_attributes(G, l, 'label')
    print("load graph", G.number_of_nodes(), G.number_of_edges())
    return G

def run_target_model(method, input_filename, output_dir, embedding_test_dir, **kargs):
    sys.path.append(embedding_test_dir)
    from src.baseline import baseline
    with cd(embedding_test_dir):
        baseline(method, None, kargs['emd_size'], input_filename, output_dir, **kargs)

def run_test(task, dataset_name, models, labels, save_filename, embedding_test_dir):
    sys.path.append(embedding_test_dir)
    from src.test import test
    args = {'radio': [0.7]}
    args['label_name'] = labels
    with cd(embedding_test_dir):
        test(task, None, dataset_name, models, save_filename=save_filename, **args)

def get_names(method, **args):
    if method == 'node2vec':
        kargs = {'emd_size': 128, 'num-walks': 10, 'walk-length': 80, 'window-size': 10, 'p': args['p'], 'q': args['q']}
        embedding_filename = os.path.join("{}_{:d}_{:d}_{:d}_{:d}_{:.4f}_{:.4f}".format(method, kargs['emd_size'], kargs['num-walks'], kargs['walk-length'], kargs['window-size'], kargs['p'], kargs['q']))
        return embedding_filename

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class GaussianProcessRegressor(object):
    def __init__(self):
        self.gp = gaussian_process.GaussianProcessRegressor()

    def fit(self, X, y):
        self.gp.fit(X, y)

    def predict(self, ps, p_bound, w=None, num=100):
        X = []
        for k in zip(*[np.linspace(p_bound[i][0], p_bound[i][1], num=num) for i in ps]):
            X.append(k)
        if w is not None:
            X = np.hstack((X, np.tile(w, (len(X), 1))))
        y = self.gp.predict(X)
        ind = np.argmax(y)
        return X[ind], y[ind]
