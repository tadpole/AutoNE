import os, sys
import random
import itertools

import numpy as np
import networkx as nx
import netlsd
from sklearn import gaussian_process

import utils

embedding_test_dir = 'embedding_test'

def sample_graph(G, output_dir, times=10):
    s_n = int(np.sqrt(G.number_of_nodes()))
    for t in range(times):
        t_dir = os.path.join(output_dir, 's{}'.format(t))
        n = random.randint(int(s_n/2), 2*s_n)
        Gs = utils.random_walk_induced_graph_sampling(G, n)
        mapping = dict(zip(Gs.nodes(), range(Gs.number_of_nodes())))
        Gs = nx.relabel_nodes(Gs, mapping)
        file_path = os.path.join(t_dir, 'graph.edgelist')
        label_path = os.path.join(t_dir, 'label.txt')
        print("sample graph, nodes: {}, edges: {}, save into {}".format(Gs.number_of_nodes(), Gs.number_of_edges(), t_dir))
        with utils.write_with_create(file_path) as f:
            for i, j in Gs.edges():
                print(i, j, file=f)
        with utils.write_with_create(label_path) as f:
            for i, data in Gs.nodes(data=True):
                if 'label' in data:
                    for j in data['label']:
                        print(i, j, file=f)

def get_result(dataset_name, target_model, task, kargs, sampled_dir='', cache=True):
    embedding_filename = utils.get_names(target_model, **kargs)
    cf = os.path.abspath(os.path.join('result/{}'.format(dataset_name), sampled_dir, 'cf', embedding_filename))
    embedding_filename = os.path.abspath(os.path.join('embeddings/{}'.format(dataset_name), sampled_dir, embedding_filename))
    dataset_filename = os.path.abspath(os.path.join('data/{}'.format(dataset_name), sampled_dir, 'graph.edgelist'))
    if (not cache) or (not os.path.exists(embedding_filename)) or (os.path.getmtime(embedding_filename) < os.path.getmtime(dataset_filename)):
        utils.run_target_model(target_model, dataset_filename, os.path.dirname(embedding_filename), embedding_test_dir=embedding_test_dir, **kargs)
    if (not cache) or (not os.path.exists(cf)) or (os.path.getmtime(cf) < os.path.getmtime(embedding_filename)):
        labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename), 'label.txt'))
        utils.run_test(task, dataset_name, [embedding_filename], labels, cf, embedding_test_dir=embedding_test_dir)
    return np.loadtxt(cf, dtype=float)

def get_wne(dataset_name, sampled_dir='', cache=True):
    dataset_filename = os.path.abspath(os.path.join('data/{}'.format(dataset_name), sampled_dir, 'graph.edgelist'))
    labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename), 'label.txt'))
    save_path = os.path.abspath(os.path.join('embeddings/{}'.format(dataset_name), sampled_dir, 'wme.embeddings'))
    if (not cache) or (not os.path.exists(save_path)) or (os.path.getmtime(save_path) < os.path.getmtime(dataset_filename)):
        G = utils.load_graph(dataset_filename, labels)
        wne = netlsd.heat(G, timescales=np.logspace(-2, 2, 10))
        with utils.write_with_create(save_path) as f:
            print(" ".join(map(str, wne)), file=f)
    return np.loadtxt(save_path)

def mle(dataset_name, target_model, task='classification', sampled_number=10):
    params = {'emd_size': [128],
              'p': utils.rand(4, 0.0001, 2),
              'q': utils.rand(4, 0.0001, 2),
              'num-walks': [10],
              'walk-length': [80],
              'window-size': [10]}
    X = []
    y = []
    ps = ['p', 'q']
    p_bound = {'p': [0.0001, 2],
               'q': [0.0001, 2]}
    for i in range(sampled_number):
        wne = get_wne(dataset_name, 'sampled/s{}'.format(i), cache=True)
        for v in itertools.product(*params.values()):
            kargs = dict(zip(params.keys(), v))
            res = get_result(dataset_name, target_model, task, kargs, 'sampled/s{}'.format(i))
            X.append(np.hstack(([kargs[p] for p in ps], wne)))
            y.append(res)
    X = np.vstack(X)
    y = np.vstack(y)
    gp = utils.GaussianProcessRegressor()
    gp.fit(X, y[:, 0])

    wne = get_wne(dataset_name, '', cache=True)
    X_b, y_b = gp.predict(ps, p_bound, wne)
    print("##################################################")
    print("best params, ", X_b, y_b)

    args = {'emd_size': 128,
              'p': X_b[0],
              'q': X_b[1],
              'num-walks': 10,
              'walk-length': 80,
              'window-size': 10}
    res = get_result(dataset_name, target_model, task, args, '')
    print("real acc, ", res)
    print("##################################################")

def mle_large(dataset_name, target_model, task):
    params = {'emd_size': [128],
              'p': [0.0001, 0.5, 1, 2],
              'q': [0.0001, 0.5, 1, 2],
              'num-walks': [10],
              'walk-length': [80],
              'window-size': [10]}
    X = []
    y = []
    ps = ['p', 'q']
    p_bound = {'p': [0.0001, 2],
               'q': [0.0001, 2]}
    for v in itertools.product(*params.values()):
        kargs = dict(zip(params.keys(), v))
        res = get_result(dataset_name, target_model, task, kargs, '')
        X.append([kargs[p] for p in ps])
        y.append(res)
    X = np.array(X)
    y = np.array(y)
    gp = utils.GaussianProcessRegressor()
    gp.fit(X, y[:, 0])
    X_b, y_b = gp.predict(ps, p_bound, None)
    print("##################################################")
    print("best params, ", X_b, y_b)

    args = {'emd_size': 128,
              'p': X_b[0],
              'q': X_b[1],
              'num-walks': 10,
              'walk-length': 80,
              'window-size': 10}
    res = get_result(dataset_name, target_model, task, args, '')
    print("real acc, ", res)
    print("##################################################")

def grid_search(dataset_name, target_model, task):
    params = {'emd_size': [128],
              'p': [0.0001, 0.5, 1, 2],
              'q': [0.0001, 0.5, 1, 2],
              'num-walks': [10],
              'walk-length': [80],
              'window-size': [10]}
    X = []
    y = []
    ps = ['p', 'q']
    for v in itertools.product(*params.values()):
        kargs = dict(zip(params.keys(), v))
        res = get_result(dataset_name, target_model, task, kargs, '')
        X.append([kargs[p] for p in ps])
        y.append(res)
    X = np.array(X)
    y = np.array(y)
    for i in zip(X, y):
        print(i)
    ind = np.argmax(y[:, 0])
    print("##################################################")
    print("best params, ", X[ind], y[ind])
    print("##################################################")

def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    dataset_name = 'BlogCatalog'
    target_model = 'node2vec'
    task = 'classification'
    dataset_path = 'data/{}/graph.edgelist'.format(dataset_name)
    label_path = 'data/{}/label.txt'.format(dataset_name)
    G = utils.load_graph(dataset_path, label_path)
    sampled_number = int(np.sqrt(G.number_of_nodes()))
    #sample_graph(G, 'data/{}/sampled'.format(dataset_name), times=sampled_number)
    mle(dataset_name, target_model, task)
    #mle_large(dataset_name, target_model, task)
    #grid_search(dataset_name, target_model, task)

if __name__ == '__main__':
    main()
