import os, sys
import random
import itertools

import numpy as np
import networkx as nx
import netlsd
from sklearn import gaussian_process
from bayes_opt import BayesianOptimization

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
    rs = utils.RandomState()
    rs.save_state()
    rs.set_seed(0)
    embedding_filename = utils.get_names(target_model, **kargs)
    cf = os.path.abspath(os.path.join('result/{}'.format(dataset_name), sampled_dir, 'cf', embedding_filename))
    embedding_filename = os.path.abspath(os.path.join('embeddings/{}'.format(dataset_name), sampled_dir, embedding_filename))
    dataset_filename = os.path.abspath(os.path.join('data/{}'.format(dataset_name), sampled_dir, 'graph.edgelist'))
    if (not cache) or (not os.path.exists(embedding_filename)) or (os.path.getmtime(embedding_filename) < os.path.getmtime(dataset_filename)):
        utils.run_target_model(target_model, dataset_filename, os.path.dirname(embedding_filename), embedding_test_dir=embedding_test_dir, **kargs)
    if (not cache) or (not os.path.exists(cf)) or (os.path.getmtime(cf) < os.path.getmtime(embedding_filename)):
        labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename), 'label.txt'))
        utils.run_test(task, dataset_name, [embedding_filename], labels, cf, embedding_test_dir=embedding_test_dir)
    rs.load_state()
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

def mle(dataset_name, target_model, task='classification', sampled_number=10, without_wne=False, k=16):
    X = []
    y = []
    ps = ['num-walks', 'walk-length', 'window-size', 'p', 'q']
    params = utils.Params(target_model)
    for i in range(sampled_number):
        wne = get_wne(dataset_name, 'sampled/s{}'.format(i), cache=True)
        for v in range(k):
            kargs = params.random_args(ps)
            print(kargs)
            res = get_result(dataset_name, target_model, task, kargs, 'sampled/s{}'.format(i))
            if without_wne:
                X.append([kargs[p] for p in ps])
            else:
                X.append(np.hstack(([kargs[p] for p in ps], wne)))
            y.append(res)
    X = np.vstack(X)
    y = np.vstack(y)
    gp = utils.GaussianProcessRegressor()
    gp.fit(X, y[:, 0])

    wne = get_wne(dataset_name, '', cache=True) if not without_wne else None
    X_b, y_b = gp.predict(ps, params.get_bound(ps), params.get_type(ps), wne)
    print("##################################################")
    X_b = params.convert(X_b, ps)
    print("best params, ", X_b, y_b)

    args = params.random_args(ps=ps, known_args=dict(zip(ps, X_b)))
    res = get_result(dataset_name, target_model, task, args, '')
    print("real acc, ", res)
    print("##################################################")
    return X_b, res

def grid_search(dataset_name, target_model, task, s=4):
    params = {'emd_size': [128],
              'p': np.linspace(0.0001, 2, s),
              'q': np.linspace(0.0001, 2, s),
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

def random_search(dataset_name, target_model, task, k=16):
    X = []
    y = []
    ps = ['num-walks', 'walk-length', 'window-size', 'p', 'q']
    params = utils.Params(target_model)
    for v in range(k):
        kargs = params.random_args(ps)
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
    return X[ind], y[ind]


def b_opt(dataset_name, target_model, task, k=16):
    ps = ['num-walks', 'walk-length', 'window-size', 'p', 'q']
    params = utils.Params(target_model)
    p_bound = dict(zip(ps, params.get_bound(ps)))
    def black_box_function(**kargs):
        x = [kargs[p] for p in ps]
        args = params.convert(x, ps)
        kargs = dict(zip(ps, args))
        kargs['emd_size'] = 128
        return get_result(dataset_name, target_model, task, kargs, '')[0]
    opt = BayesianOptimization(
            f=black_box_function,
            pbounds=p_bound,
            verbose=2)
    opt.maximize(init_points=0, n_iter=k)
    X = [opt.max['params'][p] for p in ps]
    y = opt.max['target']
    print(opt.max)
    return X, y

def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    dataset_name = 'cora'#'BlogCatalog'
    target_model = 'node2vec'
    task = 'classification'
    dataset_path = 'data/{}/graph.edgelist'.format(dataset_name)
    label_path = 'data/{}/label.txt'.format(dataset_name)
    G = utils.load_graph(dataset_path, label_path)
    sampled_number = int(np.sqrt(G.number_of_nodes()))
    m = 'b_opt'
    #sample_graph(G, 'data/{}/sampled'.format(dataset_name), times=sampled_number)
    k = 4
    for m in ['mle', 'mle_w', 'random_search', 'b_opt']:
        for k in range(1, 6):
            Xs, ys = [], []
            for i in range(5):
                if m == 'mle':
                    X, y = mle(dataset_name, target_model, task, sampled_number, without_wne=False, k=k)
                elif m == 'mle_w':
                    X, y = mle(dataset_name, target_model, task, sampled_number, without_wne=True, k=k)
                elif m == 'random_search':
                    X, y = random_search(dataset_name, target_model, task, k=k)
                elif m == 'b_opt':
                    X, y = b_opt(dataset_name, target_model, task, k=k)
                Xs.append(X)
                ys.append(y)
            Xs = np.array(Xs)
            ys = np.array(ys)
            print(Xs, ys)
            #random_search(dataset_name, target_model, task)
            #b_opt(dataset_name, target_model, task)
            save_filename = 'result/{}/{}_{}_{}.npz'.format(dataset_name, m, target_model, k)
            np.savez(save_filename, X=Xs, y=ys)

if __name__ == '__main__':
    main()
