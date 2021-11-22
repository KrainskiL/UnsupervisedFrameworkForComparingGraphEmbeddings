import networkx as nx
from node2vec import Node2Vec

g = nx.read_edgelist('edge.dat')
node2vec = Node2Vec(g, dimensions=32, p=1, q=1, quiet=True, workers=4, seed=42)
model = node2vec.fit()
model.wv.save_word2vec_format("n2v-abcd-p1-32")

import os
import subprocess
import xgboost as xgb
import igraph as ig
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import choice
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as AMI, roc_auc_score
from sklearn.cluster import KMeans

# produce rewired graphs

for i in [0.2,0.4,0.6,0.8,1.0]:
    os.system(f'julia ./shuffle_edges.jl edge.dat com.dat n2v-abcd-p1-32 edge{int(i*10)}n2v32.dat {i}')

header = "graph,emb,best_alpha,best_div,best_div_ext,best_div_int,best_alpha_auc,best_auc,best_auc_err,acc,ami,auc\n"
y = np.loadtxt('com.dat',dtype='uint16',usecols=(0))

with open(f"results_rewiring.csv","w") as fcsv:
    fcsv.write(header)
    for graph in ['edge','edge2','edge4','edge6','edge8','edge10']:
        suf = '' if graph == 'edge' else 'n2v32'
        out = subprocess.check_output(f'julia ../CGE_CLI.jl -g {graph}{suf}.dat -c com.dat -e n2v-abcd-p1-32 --force-exact --seed 42 -l 1500',shell=True)
        out = eval(out.decode('utf-8'))

        #Comm detect
        X = pd.read_csv('n2v-abcd-p1-32', sep=' ', header=None)
        X = X.dropna(axis=1)
        X = X.set_index(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Train XGBoost Classifier
        model = xgb.XGBClassifier(objective='multi:softmax', random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = sum(y_test == pred)/y_test.shape[0]

        #Clustering
        kmeans = KMeans(n_clusters=64, random_state=42).fit(X)
        pred = kmeans.labels_
        ami = AMI(y,pred)

        #Linkpred
        G = ig.Graph.Read_Edgelist(graph+suf+'.dat')
        test_size = int(np.round(.1*G.ecount()))
        np.random.seed(42)
        test_eid = np.random.choice(G.ecount(), size=test_size, replace=False)
        train_eid = np.array(list(set(range(G.ecount())).difference(test_eid)))
        test_edges = [(G.es[eid].source, G.es[eid].target) for eid in test_eid]
        train_edges = [(G.es[eid].source, G.es[eid].target) for eid in train_eid]
        size = G.ecount()
        np.random.seed(42)
        non_edges = [tuple(choice(G.vcount(),size=2,replace=False)) for i in range(3*size)]
        non_edges = [x for x in non_edges if G.get_eid(x[0],x[1],directed=True,error=False) == -1]
        non_edges = np.array(list(set(non_edges)))
        train_non_edges = non_edges[:len(train_edges)]
        test_non_edges = non_edges[len(train_edges):len(train_edges)+len(test_edges)]
        F=[]
        for s,e in train_edges:
            src_e = np.array(X.loc[s])
            dst_e = np.array(X.loc[e])
            F.append(np.concatenate((src_e, dst_e), axis=0))
        f = [1]*len(train_edges)
        for s,e in train_non_edges:
            src_e = np.array(X.loc[s])
            dst_e = np.array(X.loc[e])
            F.append(np.concatenate((src_e, dst_e), axis=0))
        F = np.array(F)
        f.extend([0]*len(train_non_edges))

        model = xgb.XGBClassifier(random_state=42)
        model.fit(F, f)

        F_test=[]
        for s,e in test_edges:
            src_e = np.array(X.loc[s])
            dst_e = np.array(X.loc[e])
            F_test.append(np.concatenate((src_e, dst_e), axis=0))
        f_test = [1]*len(test_edges)
        for s,e in test_non_edges:
            src_e = np.array(X.loc[s])
            dst_e = np.array(X.loc[e])
            F_test.append(np.concatenate((src_e, dst_e), axis=0))
        F_test = np.array(F_test)
        f_test.extend([0]*len(test_non_edges))

        pred = model.predict_proba(F_test)[:,1]
        auc = roc_auc_score(f_test, pred)
        line = f"{graph},n2v-abcd-p1-32,{out[0]},{out[1]},{out[2]},{out[3]},{out[4]},{out[5]},{out[6]},{str(acc)},{str(ami)},{str(auc)}\n"
        fcsv.write(line)
        print(line)

for i in [1.1,1.2,1.3,1.4,1.5]:
    os.system(f'julia scale_communities.jl com.dat n2v-abcd-p1-32 n2v-abcd-p1-32-{int(i*100)} {i}')

header = "graph,emb,best_alpha,best_div,best_div_ext,best_div_int,best_alpha_auc,best_auc,best_auc_err\n"
y = np.loadtxt('com.dat',dtype='uint16',usecols=(0))

with open(f"results_rescaling.csv","w") as fcsv:
    fcsv.write(header)
    for scale in ['100','110','120','130','140','150']:
        suf = '' if scale == '100' else '-'+scale
        out = subprocess.check_output(f'julia ../CGE_CLI.jl -g edge0 -c com.dat -e n2v-abcd-p1-32{suf} --force-exact --seed 42',shell=True)
        out = eval(out.decode('utf-8'))

        line = f"edge.dat,n2v-abcd-p1-32{suf},{out[0]},{out[1]},{out[2]},{out[3]},{out[4]},{out[5]},{out[6]}\n"
        fcsv.write(line)
        print(line)
