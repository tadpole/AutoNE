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
from scipy import sparse
from bayes_opt import BayesianOptimization

import utils

embedding_test_dir = 'embedding_test'
debug = True
cache = True

def split_graph(G, output_dir, radio=0.8):
    t_dir = output_dir
    Gs = G
    file_path = os.path.join(t_dir, 'graph.edgelist')
    file_test_path = os.path.join(t_dir, 'graph_test.edgelist')
    label_path = os.path.join(t_dir, 'label.txt')
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
    print(len(nodes), Gs.number_of_nodes())
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

def sample_graph(G, output_dir, s_n, times=10, with_test=False, radio=0.8, feature_path=None):
    if s_n is None:
        s_n = int(np.sqrt(G.number_of_nodes()))
    for t in range(times):
        t_dir = os.path.join(output_dir, 's{}'.format(t))
        n = random.randint(int(s_n/2), 2*s_n)
        Gs = utils.random_walk_induced_graph_sampling(G, n)
        mapping = dict(zip(Gs.nodes(), range(Gs.number_of_nodes())))
        if feature_path is not None:
            feats = sparse.load_npz(feature_path)
            row = []
            col = []
            data = []
            fr, fc = feats.nonzero()
            for i, j in zip(fr, fc):
                if i in mapping:
                    row.append(mapping[i])
                    col.append(j)
                    data.append(feats[i, j])
            feats = sparse.csr_matrix((data, (row, col)), shape=(len(mapping), feats.shape[1]))
        Gs = nx.relabel_nodes(Gs, mapping)
        file_path = os.path.join(t_dir, 'graph.edgelist')
        file_test_path = os.path.join(t_dir, 'graph_test.edgelist')
        label_path = os.path.join(t_dir, 'label.txt')
        feature_save_path = os.path.join(t_dir, 'features.npz')
        if feature_path is not None:
            utils.write_with_create(feature_save_path)
            sparse.save_npz(feature_save_path, feats)
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


def get_result(dataset_name, target_model, task, kargs, sampled_dir='', debug=debug, cache=cache):
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
    if target_model != 'gcn':
        if (not cache) or (not os.path.exists(embedding_filename)) or (os.path.getmtime(embedding_filename) < os.path.getmtime(dataset_filename)):
            utils.run_target_model(target_model, dataset_filename, os.path.dirname(embedding_filename), embedding_test_dir=embedding_test_dir, debug=debug, **kargs)
        if (not cache) or (not os.path.exists(cf)) or (os.path.getmtime(cf) < os.path.getmtime(embedding_filename)):
            if task == 'classification':
                labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename), 'label.txt'))
            elif task == 'link_predict':
                labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename)))
            utils.run_test(task, dataset_name, [embedding_filename], labels, cf, embedding_test_dir=embedding_test_dir)
    else:
        if (not cache) or (not os.path.exists(cf)):
            data_path = os.path.abspath(os.path.join('data/{}'.format(dataset_name)))
            with utils.cd(os.path.join(embedding_test_dir, 'src/baseline/gcn/gcn')):
                cmd = ('python3 main.py' +\
                        ' --epochs {} --hidden1 {} --learning_rate {}' +\
                        ' --output_filename {} --debug {} --dataset {} --input_dir {}').format(kargs['epochs'], kargs['hidden1'], kargs['learning_rate'], cf, debug, dataset_name, data_path)
                if debug:
                    print(cmd)
                else:
                    cmd += ' > /dev/null 2>&1'
                os.system(cmd)
    rs.load_state()
    res = np.loadtxt(cf, dtype=float)
    if len(res.shape) != 0:
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

def mle_k(dataset_name, target_model, task='classification', sampled_number=10, without_wne=False, k=16, s=0, print_iter=10, debug=False):
    X = []
    y = []
    params = utils.Params(target_model)
    ps = params.arg_names
    total_t = 0.0
    info = []
    X_t, res_t = None, -1.0
    if without_wne:
        gp = utils.GaussianProcessRegressor()
    else:
        K = utils.K(len(ps))
        gp = utils.GaussianProcessRegressor(K)
    for t in range(sampled_number):
        b_t = time.time()
        i = t
        wne = get_wne(dataset_name, 'sampled/s{}'.format(i), cache=True)
        for v in range(k):
            kargs = params.random_args(ps)
            res = get_result(dataset_name, target_model, task, kargs, 'sampled/s{}'.format(i))
            if without_wne:
                X.append([kargs[p] for p in ps])
            else:
                X.append(np.hstack(([kargs[p] for p in ps], wne)))
            if debug:
                print('sample {}, {}/{}, kargs: {}, res: {}, time: {:.4f}s'.format(t, v, k, [kargs[p] for p in ps], res, time.time()-b_t))
            y.append(res)

    for t in range(s):
        b_t = time.time()
        gp.fit(np.vstack(X), y)
        X_temp, res_temp = _get_mle_result(gp, dataset_name, target_model, task, without_wne, params, ps, 0, X, y)
        if without_wne:
            X.append(X_temp)
        else:
            X.append(np.hstack((X_temp, wne)))
        y.append(res_temp)
        if res_t < res_temp:
            res_t = res_temp
            X_t = X_temp
        e_t = time.time()
        total_t += e_t-b_t
        info.append([res_temp, total_t])
        print('iters: {}/{}, params: {}, res: {}, time: {:.4f}s'.format(t, s, X_temp, res_temp, total_t))
    if debug:
        return X_t, res_t, info
    return X_t, res_t

def random_search(dataset_name, target_model, task, k=16, debug=False, sampled_dir=''):
    X = []
    y = []
    params = utils.Params(target_model)
    ps = params.arg_names
    b_t = time.time()
    info = []
    for v in range(k):
        kargs = params.random_args(ps)
        #kargs = params.convert_dict(kargs, ps)
        if debug:
            print(kargs)
        res = get_result(dataset_name, target_model, task, kargs, sampled_dir)
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


def b_opt(dataset_name, target_model, task, k=16, debug=False, n_inits=0, inits=None, sampled_dir=''):
    params = utils.Params(target_model)
    ps = params.arg_names
    p_bound = dict(zip(ps, params.get_bound(ps)))
    def black_box_function(**kargs):
        b_t = time.time()
        x = [kargs[p] for p in ps]
        args = params.convert(x, ps)
        kargs = dict(zip(ps, args))
        kargs['emd_size'] = 64
        if target_model == 'AROPE':
            kargs['order'] = 3
        res = get_result(dataset_name, target_model, task, kargs, sampled_dir)
        e_t = time.time()
        print("############## params: {}, time: {}s".format(kargs, e_t-b_t))
        return res
    opt = BayesianOptimization(
            f=black_box_function,
            pbounds=p_bound,
            verbose=2)
    #opt.set_gp_params(normalize_y=False)
    if inits is not None:
        for d in inits:
            dd = dict(zip(ps, d))
            target = black_box_function(**dd)
            print(dd, target)
            opt.register(params=dd, target=target)
    opt.maximize(init_points=n_inits, n_iter=k)
    X = [opt.max['params'][p] for p in ps]
    y = opt.max['target']
    if debug:
        info = [res['target'] for res in opt.res]
        return X, y, info
    return X, y

def test_1(dataset_name, target_model, task):
    params = utils.Params(target_model)
    ps = params.arg_names
    b_t = time.time()
    info = []
    sampled_dir = 'sampled/s0'
    X = []
    y = []
    args = {'number-walks': 10, 'walk-length': 10, 'window-size': 3}
    temp_args = params.random_args(ps)
    res = get_result(dataset_name, target_model, task, temp_args, sampled_dir, cache=True)
    X.append([temp_args[p] for p in ps])
    y.append(res)
    print(i, j, [temp_args[p] for p in ps], res)
    return 0

def main(args):
    seed = None
    random.seed(seed)
    np.random.seed(seed)
    if len(args) == 0:
        dataset_name = 'pubmed'
        target_model = 'gcn'
        task = 'classification'
        ms = ['mle', 'random_search', 'b_opt']
    else:
        dataset_name = args[0]
        target_model = args[1]
        task = args[2]
        ms = args[3:]
    with_test = False
    dataset_path = 'data/{}/graph.edgelist'.format(dataset_name)
    label_path = 'data/{}/label.txt'.format(dataset_name)
    feature_path = None
    if task == 'link_predict':
        dataset_name = dataset_name+'_0.8'
        label_path = None
        with_test = True
    if target_model == 'gcn':
        feature_path = 'data/{}/features.npz'.format(dataset_name)
    if target_model == 'sample':
        G = utils.load_graph(dataset_path, label_path)
        split_graph(G, 'data/{}_0.8'.format(dataset_name), radio=0.8)
        sampled_number = 10#int(np.sqrt(G.number_of_nodes()))
        sample_graph(G, 'data/{}/sampled'.format(dataset_name), s_n=1000, times=5, with_test=with_test, feature_path=feature_path)
        return 0
    ks = 5
    #test(dataset_name, target_model, task)
    sampled_dir = ''

    for m in ms:
        res = []
        for i in range(ks):
            info = []
            if m == 'mle':
                X, y, info = mle_k(dataset_name, target_model, task, sampled_number=5, without_wne=False, k=5, s=10, debug=True)
            elif m == 'mle_w':
                X, y, info = mle_k(dataset_name, target_model, task, sampled_number=5, without_wne=True, k=5, s=10, debug=True)
            elif m == 'random_search':
                X, y, info = random_search(dataset_name, target_model, task, k=10, debug=True, sampled_dir=sampled_dir)
            elif m == 'random_search_l':
                X, y, info = random_search(dataset_name, target_model, task, k=5, debug=True, sampled_dir=sampled_dir)
            elif m == 'b_opt':
                b_t = time.time()
                X, y, info_t = b_opt(dataset_name, target_model, task, k=5, n_inits=5, debug=True, sampled_dir=sampled_dir)
                e_t = time.time()
                info = [[j, (e_t-b_t)/len(info_t)*(i+1)] for i, j in enumerate(info_t)]
            elif m == 'b_opt_l':
                b_t = time.time()
                X, y, info_t = b_opt(dataset_name, target_model, task, k=5, n_inits=1, debug=True, sampled_dir=sampled_dir)
                e_t = time.time()
                info = [[j, (e_t-b_t)/len(info_t)*(i+1)] for i, j in enumerate(info_t)]
            res.append(info)
            print(m, i, res)
            ts = 'lp' if task == 'link_predict' else 'cf'
            if sampled_dir == '':
                save_filename = 'result/{}/res_{}_{}_{}.npz'.format(dataset_name, ts, m, target_model)
            else:
                save_filename = 'result/{}/res_{}_{}_{}_{}.npz'.format(dataset_name, os.path.basename(sampled_dir), ts, m, target_model)
            np.savez(save_filename, res=res)

if __name__ == '__main__':
    main(sys.argv[1:])
