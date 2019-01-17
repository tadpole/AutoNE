import os, sys
import random
import itertools
import time
import copy
import functools

import numpy as np
import networkx as nx
import netlsd
from sklearn import gaussian_process
from bayes_opt import BayesianOptimization

import utils

embedding_test_dir = 'embedding_test'

def sample_graph(G, output_dir, times=10, with_test=False, radio=0.8):
    s_n = int(np.sqrt(G.number_of_nodes()))
    for t in range(times):
        t_dir = os.path.join(output_dir, 's{}'.format(t))
        n = random.randint(int(s_n/2), 2*s_n)
        Gs = utils.random_walk_induced_graph_sampling(G, n)
        mapping = dict(zip(Gs.nodes(), range(Gs.number_of_nodes())))
        Gs = nx.relabel_nodes(Gs, mapping)
        file_path = os.path.join(t_dir, 'graph.edgelist')
        file_test_path = os.path.join(t_dir, 'graph_test.edgelist')
        label_path = os.path.join(t_dir, 'label.txt')
        if not with_test:
            print("sample graph, nodes: {}, edges: {}, save into {}".format(Gs.number_of_nodes(), Gs.number_of_edges(), t_dir))
            with utils.write_with_create(file_path) as f:
                for i, j in Gs.edges():
                    print(i, j, file=f)
            with utils.write_with_create(label_path) as f:
                for i, data in Gs.nodes(data=True):
                    if 'label' in data:
                        for j in data['label']:
                            print(i, j, file=f)
        else:
            G_train = nx.Graph()
            G_test = nx.Graph()
            edges = np.random.permutation(list(Gs.edges()))
            nodes = set()
            for a, b in edges:
                if a not in nodes or b not in nodes:
                    G_train.add_edge(a, b)
                    nodes.add(a)
                    nodes.add(b)
                else:
                    G_test.add_edge(a, b)
            assert len(nodes) == Gs.number_of_nodes()
            assert len(nodes) == G_train.number_of_nodes()
            num_test_edges = int((1-radio)*Gs.number_of_edges())
            now_number = G_test.number_of_edges()
            if num_test_edges < now_number:
                test_edges = list(G_test.edges())
                G_train.add_edges_from(test_edges[:now_number-num_test_edges])
                G_test.remove_edges_from(test_edges[:now_number-num_test_edges])
            print("sample graph,origin: {} {}, train: {} {}, test: {} {}".format(Gs.number_of_nodes(), Gs.number_of_edges(), G_train.number_of_nodes(), G_train.number_of_edges(), G_test.number_of_nodes(), G_test.number_of_edges()))
            with utils.write_with_create(file_path) as f:
                for i, j in G_train.edges():
                    print(i, j, file=f)
            with utils.write_with_create(file_test_path) as f:
                for i, j in G_test.edges():
                    print(i, j, file=f)


def get_result(dataset_name, target_model, task, kargs, sampled_dir='', debug=False, cache=True):
    rs = utils.RandomState()
    rs.save_state()
    rs.set_seed(0)
    embedding_filename = utils.get_names(target_model, **kargs)
    if task == 'classification':
        cf = os.path.abspath(os.path.join('result/{}'.format(dataset_name), sampled_dir, 'cf', embedding_filename))
    elif task == 'link_predict':
        cf = os.path.abspath(os.path.join('result/{}'.format(dataset_name), sampled_dir, 'lp', embedding_filename))
    embedding_filename = os.path.abspath(os.path.join('embeddings/{}'.format(dataset_name), sampled_dir, embedding_filename))
    dataset_filename = os.path.abspath(os.path.join('data/{}'.format(dataset_name), sampled_dir, 'graph.edgelist'))
    if (not cache) or (not os.path.exists(embedding_filename)) or (os.path.getmtime(embedding_filename) < os.path.getmtime(dataset_filename)):
        utils.run_target_model(target_model, dataset_filename, os.path.dirname(embedding_filename), embedding_test_dir=embedding_test_dir, debug=debug, **kargs)
    if (not cache) or (not os.path.exists(cf)) or (os.path.getmtime(cf) < os.path.getmtime(embedding_filename)):
        if task == 'classification':
            labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename), 'label.txt'))
        elif task == 'link_predict':
            labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename)))
        utils.run_test(task, dataset_name, [embedding_filename], labels, cf, embedding_test_dir=embedding_test_dir)
    rs.load_state()
    res = np.loadtxt(cf, dtype=float)
    if task == 'classification':
        res = res[0]
    return res

def get_wne(dataset_name, sampled_dir='', cache=True):
    dataset_filename = os.path.abspath(os.path.join('data/{}'.format(dataset_name), sampled_dir, 'graph.edgelist'))
    labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename), 'label.txt'))
    save_path = os.path.abspath(os.path.join('embeddings/{}'.format(dataset_name), sampled_dir, 'wme.embeddings'))
    if (not cache) or (not os.path.exists(save_path)) or (os.path.getmtime(save_path) < os.path.getmtime(dataset_filename)):
        G = utils.load_graph(dataset_filename, label_name=None)
        do_full = (G.number_of_nodes()<10000)
        eigenvalues = 'full' if do_full else 'auto'
        wne = netlsd.heat(G, timescales=np.logspace(-2, 2, 10), eigenvalues=eigenvalues)
        with utils.write_with_create(save_path) as f:
            print(" ".join(map(str, wne)), file=f)
    return np.loadtxt(save_path)

def _get_mle_result(gp, dataset_name, target_model, task, without_wne, params, ps, s, X, y):
    wne = get_wne(dataset_name, '', cache=True) if not without_wne else None
    X_b_t, res_t = None, -1.0
    X_t = copy.deepcopy(X)
    y_t = copy.deepcopy(y)
    for i in range(s):
        X_b, y_b = gp.predict(ps, params.get_bound(ps), params.get_type(ps), wne)
        X_b = params.convert(X_b, ps)

        args = params.random_args(ps=ps, known_args=dict(zip(ps, X_b)))
        res = get_result(dataset_name, target_model, task, args, '')
        if res_t < res:
            res_t = res
            X_b_t = X_b
        if without_wne:
            X_b = [X_b]
        else:
            X_b = np.hstack((X_b, wne))
        X_t = np.vstack((X_t, X_b))
        y_t.append(res)
        gp.fit(X_t, y_t)
    X_b, y_b = gp.predict(ps, params.get_bound(ps), params.get_type(ps), wne)
    X_b = params.convert(X_b, ps)

    args = params.random_args(ps=ps, known_args=dict(zip(ps, X_b)))
    res = get_result(dataset_name, target_model, task, args, '')
    if res_t < res:
        res_t = res
        X_b_t = X_b
    return X_b_t, res_t

def mle(dataset_name, target_model, task='classification', sampled_number=10, without_wne=False, k=16, s=0, times=100, print_iter=10, debug=False):
    X = []
    y = []
    params = utils.Params(target_model)
    ps = params.arg_names
    total_t = 0.0
    info = []
    for t in range(times):
        b_t = time.time()
        i = random.randint(0, sampled_number)
        wne = get_wne(dataset_name, 'sampled/s{}'.format(i), cache=True)
        for v in range(k):
            kargs = params.random_args(ps)
            res = get_result(dataset_name, target_model, task, kargs, 'sampled/s{}'.format(i))
            if without_wne:
                X.append([kargs[p] for p in ps])
            else:
                X.append(np.hstack(([kargs[p] for p in ps], wne)))
            y.append(res)
        e_t = time.time()
        total_t += e_t-b_t
        if t % print_iter == 0:
            gp = utils.GaussianProcessRegressor()
            gp.fit(np.vstack(X), y)
            X_temp, res_temp = _get_mle_result(gp, dataset_name, target_model, task, without_wne, params, ps, s, X, y)
            if debug:
                info.append([res_temp, total_t])
            print('iters: {}/{}, params: {}, res: {}, time: {:.4f}s'.format(t, times, X_temp, res_temp, total_t))
    X = np.vstack(X)
    gp = utils.GaussianProcessRegressor()
    gp.fit(X, y)
    X_b_t, res_b_t = _get_mle_result(gp, dataset_name, target_model, task, without_wne, params, ps, s, X, y)
    if debug:
        return X_b_t, res_b_t, info
    return X_b_t, res_b_t

def mle_m(dataset_name, target_model, task='classification', sampled_number=10,  k=16):
    params = utils.Params(target_model)
    ps = params.arg_names
    info = []
    gps = []
    wnes = []
    for i in range(sampled_number):
        X = []
        y = []
        wne = get_wne(dataset_name, 'sampled/s{}'.format(i), cache=True)
        for v in range(k):
            kargs = params.random_args(ps)
            res = get_result(dataset_name, target_model, task, kargs, 'sampled/s{}'.format(i))
            X.append([kargs[p] for p in ps])
            y.append(res)
        wnes.append(wne)
        gp = utils.GaussianProcessRegressor()
        gp.fit(np.vstack(X), y)
        gps.append(gp)
    wne = get_wne(dataset_name, '', cache=True)
    gp = utils.GaussianProcessRegressor()
    sims = utils.softmax([netlsd.compare(wne, w) for w in wnes])
    gp.gp.kernel = functools.reduce(lambda i, j: i+j, map(lambda x: x[0].gp.kernel_*x[1], zip(gps, sims)))
    gp.gp.kernel_ = gp.gp.kernel
    X_b_t, res_t = None, -1.0
    X_b, y_b = gp.predict(ps, params.get_bound(ps), params.get_type(ps))
    X_b = params.convert(X_b, ps)

    args = params.random_args(ps=ps, known_args=dict(zip(ps, X_b)))
    res = get_result(dataset_name, target_model, task, args, '')
    if res_t < res:
        res_t = res
        X_b_t = X_b
    return X_b_t, res_t

def random_search(dataset_name, target_model, task, k=16, debug=False):
    X = []
    y = []
    ps = ['num-walks', 'walk-length', 'window-size', 'p', 'q']
    params = utils.Params(target_model)
    b_t = time.time()
    info = []
    for v in range(k):
        kargs = params.random_args(ps)
        res = get_result(dataset_name, target_model, task, kargs, '')
        X.append([kargs[p] for p in ps])
        y.append(res)
        ind = np.argmax(y)
        total_t = time.time()-b_t
        if debug:
            info.append([y[ind], total_t])
        print('iters: {}/{}, params: {}, res: {}, time: {:.4f}s'.format(v, k, X[ind], y[ind], total_t))
    X = np.array(X)
    y = np.array(y)
    ind = np.argmax(y)
    if debug:
        return X[ind], y[ind], info
    return X[ind], y[ind]


def b_opt(dataset_name, target_model, task, k=16, debug=False, inits=None):
    ps = ['num-walks', 'walk-length', 'window-size', 'p', 'q']
    params = utils.Params(target_model)
    p_bound = dict(zip(ps, params.get_bound(ps)))
    def black_box_function(**kargs):
        x = [kargs[p] for p in ps]
        args = params.convert(x, ps)
        kargs = dict(zip(ps, args))
        kargs['emd_size'] = 128
        return get_result(dataset_name, target_model, task, kargs, '')
    opt = BayesianOptimization(
            f=black_box_function,
            pbounds=p_bound,
            verbose=2)
    if inits is not None:
        for d in inits:
            dd = dict(zip(ps, d))
            target = black_box_function(**dd)
            print(dd, target)
            opt.register(params=dd, target=target)
    opt.maximize(init_points=0, n_iter=k)
    X = [opt.max['params'][p] for p in ps]
    y = opt.max['target']
    if debug:
        info = [res['target'] for res in opt.res]
        return X, y, info
    return X, y

def main():
    #seed = 10
    #random.seed(seed)
    #np.random.seed(seed)

    dataset_name = 'citeseer'
    target_model = 'deepwalk'
    task = 'classification'#'link_predict'
    dataset_path = 'data/{}/graph.edgelist'.format(dataset_name)
    label_path = 'data/{}/label.txt'.format(dataset_name)
    with_test = False
    if task == 'link_predict':
        dataset_name = dataset_name+'_0.8'
        label_path = None
        with_test = True
    #G = utils.load_graph(dataset_path, label_path)
    #sampled_number = int(np.sqrt(G.number_of_nodes()))
    #sample_graph(G, 'data/{}/sampled'.format(dataset_name), times=sampled_number, with_test=False)
    ms = ['mle', 'random_search', 'b_opt', 'mle_w', 'mle_s']
    ms = ['mle', 'mle_m', 'random_search', 'b_opt']
    ms = ['random_search', 'b_opt']
    ms = ['mle']
    ks = 1

    for m in ms:
        res = []
        for i in range(ks):
            info = []
            if m == 'mle':
                X, y, info = mle(dataset_name, target_model, task, sampled_number=10, without_wne=False, k=1, s=0, times=300, print_iter=20, debug=True)
            elif m == 'mle_m':
                for k in [3, 5, 10, 15, 20]:
                    b_t = time.time()
                    X, y = mle_m(dataset_name, target_model, task, sampled_number=10, k=k)
                    e_t = time.time()
                    info.append([y, e_t-b_t])
            elif m == 'random_search':
                X, y, info = random_search(dataset_name, target_model, task, k=10, debug=True)
            elif m == 'b_opt':
                b_t = time.time()
                X, y, info_t = b_opt(dataset_name, target_model, task, k=10, debug=True)
                e_t = time.time()
                info = [[j, (e_t-b_t)/len(info_t)*(i+1)] for i, j in enumerate(info_t)]
            elif m == 'b_opt_o':
                b_t = time.time()
                data = np.loadtxt('temp.txt')
                X, y, info_t = b_opt(dataset_name, target_model, task, k=10, debug=True)
                e_t = time.time()
                info = [[j, (e_t-b_t)/len(info_t)*(i+1)] for i, j in enumerate(info_t)]
            res.append(info)
        print(m, res)
        save_filename = 'result/{}/res_{}_{}.npz'.format(dataset_name, m, target_model)
        np.savez(save_filename, res=res)

if __name__ == '__main__':
    main()
