import random
import os, sys
import collections
import itertools

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn import gaussian_process
from scipy.optimize import minimize

def rand(size, a, b, decimals=4):
    res = np.random.random_sample(size)*(b-a)+a
    if decimals is not None:
        return np.around(res, decimals=decimals)
    return res

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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

def run_target_model(method, input_filename, output_dir, embedding_test_dir, debug=True, **kargs):
    sys.path.append(embedding_test_dir)
    from src.baseline import baseline
    with cd(embedding_test_dir):
        baseline(method, None, kargs['emd_size'], input_filename, output_dir, debug=debug, **kargs)

def run_test(task, dataset_name, models, labels, save_filename, embedding_test_dir):
    sys.path.append(embedding_test_dir)
    from src.test import test
    args = {}
    if task == 'classification':
        args['radio'] = [0.8]
        args['label_name'] = labels
        evalution = None
    elif task == 'link_predict':
        evalution = 'AUC'
        args['data_dir'] = labels

    with cd(embedding_test_dir):
        test(task, evalution, dataset_name, models, save_filename=save_filename, **args)

def get_names(method, **args):
    if method == 'node2vec':
        kargs = args
        embedding_filename = os.path.join("{}_{:d}_{:d}_{:d}_{:d}_{:.4f}_{:.4f}".format(method, kargs['emd_size'], kargs['num-walks'], kargs['walk-length'], kargs['window-size'], kargs['p'], kargs['q']))
        return embedding_filename

def random_with_bound_type(bound, type_):
    res = []
    for b, t in zip(bound, type_):
        if t == int:
            res.append(random.randint(*b))
        elif t == float:
            res.append(rand(1, *b)[0])
        else:
            res.append(None)
    return res


def find_b_opt_max(gp, ps, p_bound, p_type, w=None, n_warmup=100000, n_iter=100):
    """
    refer to acq_max https://github.com/fmfn/BayesianOptimization/blob/master/bayes_opt/util.py
    """
    X = []
    for k in range(n_warmup):
        X.append(random_with_bound_type(p_bound, p_type))
    if w is not None:
        X = np.hstack((X, np.tile(w, (len(X), 1))))
    y = gp.predict(X)
    ind = np.argmax(y)
    x_max, y_max = X[ind][:len(ps)], y[ind]
    temp_w = [] if w is None else w
    def temp_f(x):
        return gp.predict([list(x)+list(temp_w)])[0]
    for i in range(n_iter):
        x_try = random_with_bound_type(p_bound, p_type)
        res = minimize(lambda x: -gp.predict([list(x)+list(temp_w)])[0],
                        x_try,
                        bounds=p_bound,
                        method='L-BFGS-B')
        if not res.success:
            continue
        if -res.fun >= y_max:
            x_max = res.x
            y_max = -res.fun

    return x_max, y_max

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
        self.gp = gaussian_process.GaussianProcessRegressor(
                kernel=gaussian_process.kernels.Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10)

    def fit(self, X, y):
        self.gp.fit(X, y)

    def predict(self, ps, p_bound, type_, w=None):
        return find_b_opt_max(self.gp, ps, p_bound, type_, w)

class RandomState(object):
    def __init__(self):
        self.state = None

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def save_state(self):
        self.state = (random.getstate(), np.random.get_state())

    def load_state(self):
        random.setstate(self.state[0])
        np.random.set_state(self.state[1])

class Params(object):
    def __init__(self, method):
        self.method = method
        if method == 'node2vec':
            self.arg_names = ['num-walks', 'walk-length', 'window-size', 'p', 'q']
            self.type_ = [int, int, int, float, float]
            self.bound = [(2, 20), (2, 80), (2, 10), (0.0001, 2), (0.0001, 2)]
            self.ind = dict(zip(self.arg_names, range(len(self.arg_names))))

    def get_type(self, ps=None):
        if ps is None:
            return self.type_
        return [self.type_[self.ind[p]] for p in ps]

    def get_bound(self, ps=None):
        if ps is None:
            return self.bound
        return [self.bound[self.ind[p]] for p in ps]

    def convert(self, X, ps=None):
        type_ = self.get_type(ps)
        bound = np.array(self.get_bound(ps))
        X = np.clip(X, bound[:, 0], bound[:, 1])
        res = []
        for x, t in zip(X, type_):
            if t == int:
                res.append(int(round(x, 0)))
            elif t == float:
                res.append(round(x, 4))
        return res

    def random_args(self, ps=None, emd_size=128, known_args={}):
        if ps is None:
            ps = self.arg_names
        type_ = self.get_type(ps)
        bound = self.get_bound(ps)
        res = random_with_bound_type(bound, type_)
        d = dict(zip(ps, res))
        for arg in known_args:
            d[arg] = known_args[arg]
        d['emd_size'] = emd_size
        return d

def analysis_result(data_dir):
    fs = os.listdir(data_dir)
    fs = np.array([np.loadtxt(os.path.join(data_dir, f)) for f in fs if not f.endswith('names')])
    print(fs.shape)
    scale = 100
    d = (fs[:, 0]*scale).astype(int)
    for k, v in collections.Counter(d).most_common():
        print(k*1.0/scale, v, "{:.2f}".format(v*1.0/fs.shape[0]))

if __name__ == '__main__':
    analysis_result('result/BlogCatalog/cf/')
