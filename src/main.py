import os, sys
import random
import itertools

import numpy as np
import networkx as nx

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
                    print(i, data['label'], file=f)


def run_models(method, input_filename, output_dir, embedding_test_dir=None):
    if method == 'node2vec':
        emd_size = 128
        params = {'p': [0.5, 1, 2],
                  'q': [0.5, 1, 2],
                  'num-walks': [10],
                  'walk-length': [80],
                  'window-size': [10]}
        for v in itertools.product(*params.values()):
            kargs = dict(zip(params.keys(), v))
            utils.run_target_model(method, emd_size, input_filename, output_dir, embedding_test_dir=embedding_test_dir, **kargs)

def get_result(dataset_name, target_model, task, kargs, sampled_dir='', cache=True):
    embedding_filename = utils.get_names(target_model, **kargs)
    cf = os.path.abspath(os.path.join('result/{}'.format(dataset_name), sampled_dir, 'cf', embedding_filename))
    embedding_filename = os.path.abspath(os.path.join('embeddings/{}'.format(dataset_name), sampled_dir, embedding_filename))
    dataset_filename = os.path.abspath(os.path.join('data/{}'.format(dataset_name), sampled_dir, 'graph.edgelist'))
    if (not cache) or (not os.path.exists(embedding_filename)) or (os.path.getmtime(embedding_filename) < os.path.getmtime(dataset_filename)):
        utils.run_target_model(target_model, kargs['emd_size'], dataset_filename, os.path.dirname(embedding_filename), embedding_test_dir=embedding_test_dir, **kargs)
    if (not cache) or (not os.path.exists(cf)) or (os.path.getmtime(cf) < os.path.getmtime(embedding_filename)):
        labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename), 'label.txt'))
        utils.run_test(task, dataset_name, [embedding_filename], labels, cf, embedding_test_dir=embedding_test_dir)
    return np.loadtxt(cf, dtype=float)

def mle(dataset_name, target_model, task='classification', sampled_number=10):
    params = {'emd_size': [128],
              'p': [0.5, 1, 2],
              'q': [0.5, 1, 2],
              'num-walks': [10],
              'walk-length': [80],
              'window-size': [10]}
    for i in range(sampled_number):
        for v in itertools.product(*params.values()):
            kargs = dict(zip(params.keys(), v))
            res = get_result(dataset_name, target_model, task, kargs, 'sampled/s{}'.format(i))
            print(res)

def main():
    dataset_name = 'cora'
    target_model = 'node2vec'
    task = 'classification'
    sampled_number = 10
    steps = [1, 2, 3, 4] # 0: sampling, 1: run target model, 2: test performance 3: run whole graph embedding 4: run meta learner
    mle(dataset_name, target_model, task)
    return 0
    if 0 in steps:
        dataset_path = 'data/{0}/{0}.edgelist'.format(dataset_name)
        label_path = 'data/{0}/{0}_label.txt'.format(dataset_name)
        G = utils.load_graph(dataset_path, label_path)
        sample_graph(G, 'data/{0}/sampled'.format(dataset_name), times=sampled_number)
    if 1 in steps:
        for i in range(sampled_number):
            input_filename = os.path.abspath('data/{}/sampled/s{}/graph.edgelist'.format(dataset_name, i))
            output_dir = os.path.abspath('embeddings/{}/sampled/s{}'.format(dataset_name, i))
            run_models(target_model, input_filename, output_dir, embedding_test_dir=embedding_test_dir)
    if 2 in steps:
        for i in range(sampled_number):
            embedding_dir = os.path.abspath('embeddings/{}/sampled/s{}'.format(dataset_name, i))
            save_dir = os.path.abspath('result/{}/sampled/s{}'.format(dataset_name, i))
            if task == 'classification':
                save_filename = os.path.join(save_dir, 'cf')
            models = list(map(lambda a: os.path.join(embedding_dir, a),
                                filter(lambda x: x.startswith(target_model), os.listdir(embedding_dir))))
            labels = os.path.abspath('data/{}/sampled/s{}/label.txt'.format(dataset_name, i))
            utils.run_test(task, dataset_name, models, labels, save_filename, embedding_test_dir)
    if 3 in steps:
        import netlsd
        for i in range(sampled_number):
            edgelist_filename = 'data/{}/sampled/s{}/graph.edgelist'.format(dataset_name, i)
            label_name = 'data/{}/sampled/s{}/label.txt'.format(dataset_name, i)
            save_path = 'embeddings/{}/sampled/s{}/wne.embeddings'.format(dataset_name, i)
            G = utils.load_graph(edgelist_filename, label_name)
            wne = netlsd.heat(G)
            with utils.write_with_create(save_path) as f:
                print(" ".join(map(str, wne)), file=f)
        dataset_path = 'data/{0}/{0}.edgelist'.format(dataset_name)
        label_path = 'data/{0}/{0}_label.txt'.format(dataset_name)
        save_path = 'embeddings/{}/wne.embeddings'.format(dataset_name)
        if not os.path.exists(save_path):
            G = utils.load_graph(dataset_path, label_path)
            wne = netlsd.heat(G)
            with utils.write_with_create(save_path) as f:
                print(" ".join(map(str, wne)), file=f)
    if 4 in steps:
        from sklearn import gaussian_process
        X = []
        y = []
        for i in range(sampled_number):
            data_dir = 'result/{}/sampled/s{}'.format(dataset_name, i)
            cf = np.loadtxt(os.path.join(data_dir, 'cf'), dtype=float)
            cf_name = open(os.path.join(data_dir, 'cf_names')).readlines()
            params = np.array(list(map(lambda x: x.strip().split('_')[1:], cf_name))).astype(float)
            if target_model == 'node2vec':
                params = params[:, -2:]
            wne = np.loadtxt('embeddings/{}/sampled/s{}/wne.embeddings'.format(dataset_name, i))
            feats = np.hstack((params, np.tile(wne, (len(params), 1))))
            X.append(feats)
            y.append(cf)
        X = np.vstack(X)
        y = np.vstack(y)
        gp = gaussian_process.GaussianProcessRegressor()
        gp.fit(X, y[:, 1])
        wne = np.loadtxt('embeddings/{}/wne.embeddings'.format(dataset_name, i))
        X_test = []
        print(y)
        for p in np.linspace(0, 2, num=100):
            for q in np.linspace(0, 2, num=100):
                X_test.append(np.hstack(([p, q], wne)))
        X_test = np.array(X_test)
        #print(X_test)
        y_test = gp.predict(X_test)
        print("max y", np.max(y_test))
        ind = np.argmax(y_test)
        print(X_test[ind])
    if 5 in steps:
        input_filename = os.path.abspath('data/{0}/{0}.edgelist'.format(dataset_name))
        output_dir = os.path.abspath('embeddings/{}'.format(dataset_name))
        kargs = {'p': 0.0001,
                  'q': 1.3333,
                  'num-walks': 10,
                  'walk-length': 80,
                  'window-size': 10}
        utils.run_target_model(target_model, 128, input_filename, output_dir, embedding_test_dir=embedding_test_dir, **kargs)

        embedding_dir = output_dir
        save_dir = os.path.abspath('result/{}'.format(dataset_name))
        if task == 'classification':
            save_filename = os.path.join(save_dir, 'cf')
        models = list(map(lambda a: os.path.join(embedding_dir, a),
                            filter(lambda x: x.startswith(target_model), os.listdir(embedding_dir))))
        print(models)
        labels = os.path.abspath('data/{0}/{0}_label.txt'.format(dataset_name))
        utils.run_test(task, dataset_name, models, labels, save_filename, embedding_test_dir)

if __name__ == '__main__':
    main()
