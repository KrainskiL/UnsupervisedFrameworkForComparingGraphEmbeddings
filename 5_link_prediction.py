import os
from time import time

import numpy as np
from numpy.random import choice
import pandas as pd

import igraph as ig
import networkx as nx
from node2vec import Node2Vec

import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

from common import *

datadir_graphs = 'data/graphs_reduced'
os.makedirs(datadir_graphs,exist_ok=True)

def remove_random_edges(graph:str,seed:int,input_path:str,output_path:str) -> None:
	G = ig.Graph.Read_Edgelist(f'{input_path}/{graph}.edgelist', directed=True)
	G.vs['name'] = [str(i) for i in np.arange(G.vcount())]
	test_size = int(np.round(.05*G.ecount()))
	np.random.seed(seed)
	test_eid = np.random.choice(G.ecount(), size=test_size, replace=False)
	edges = [(G.es[eid].source, G.es[eid].target) for eid in test_eid]
	with open(f'{output_path}/{graph}_s{seed}_rem_edges', "w") as f:
		for vid in edges:
			f.write(str(G.vs[vid[0]]['name'])+" "+str(G.vs[vid[1]]['name'])+"\n")
	Gp = G.copy()
	Gp.delete_edges(test_eid)
	with open(f'{output_path}/{graph}_s{seed}_reduced', "w") as f:
		for vid in Gp.es:
			f.write(str(G.vs[vid.source]['name'])+" "+str(G.vs[vid.target]['name'])+"\n")
	return None

for graph in ["sbm10k","lfr10k","nlfr10k","email"]:
    for seed in range(5):
        remove_random_edges(graph,seed,"data",datadir_graphs)

datadir_embeddings = 'data/embeddings_reduced'
os.makedirs(datadir_embeddings,exist_ok=True)

for graph in ["sbm10k","lfr10k","nlfr10k","email"]:
    for seed in range(5):
        g = ig.Graph.Read_Edgelist(f"{datadir_graphs}/{graph}_s{seed}_reduced",directed=True)
        g.vs['name'] = [str(i) for i in np.arange(g.vcount())]
        for dim in np.arange(2,33,2):
            for sim in ["ppr","katz","aa"]:
                X = Hope(g, sim = sim, dim = dim)
                fname = f'{datadir_embeddings}/hope-{graph}-{sim}-{dim}-{seed}_reduced'
                saveEmbedding(X, g, fn=fname)

for graph in ["sbm10k","lfr10k","nlfr10k","email"]:
    for seed in range(5):
        g = nx.read_edgelist(f'{datadir_graphs}/{graph}_s{seed}_reduced',create_using=nx.DiGraph())
        for dim in np.arange(2,33,2):
            for p_val in [1,0.11,9]:
                node2vec = Node2Vec(g, dimensions=dim, p=p_val, q=1/p_val, quiet=True, workers=4, seed=42)
                model = node2vec.fit()
                fname = f"{datadir_embeddings}/n2v-{graph}-p{round(p_val,2)}-{dim}-{seed}_reduced"
                model.wv.save_word2vec_format(fname)

def readEmbedding(fn="_embed", N2V=False):
    if N2V:
        D = pd.read_csv(fn, sep=' ', header=None, skiprows=1)
    else:
        D = pd.read_csv(fn, sep=' ', header=None)
    D = D.dropna(axis=1)
    D = D.set_index(0)
    return D

data_dir = "linkpred_results"
os.makedirs(data_dir,exist_ok=True)
param_trans = {"ppr":1,"katz":0.11,"aa":9}
header = "graph,emb,dim,param,exec_time,landmarks,best_alpha,best_div,best_div_ext,best_div_int,best_alpha_auc,best_auc,best_auc_err,seed,auc,auc_swap,acc,acc_swap,diff_score,std_score,auc_deg,auc_swap_deg,acc_deg,acc_swap_deg,diff_score_deg,std_score_deg\n"

for graph in ["sbm10k","lfr10k","nlfr10k","email"]:
    for seed in range(5):
        cl = f"data/{graph}.ecg"
        fullgraph = f"data/{graph}.edgelist"
        gr = f"{datadir_graphs}/{graph}_s{seed}_reduced"
        G = ig.Graph.Read_Edgelist(gr,directed=True)
        size = G.ecount()
        np.random.seed(int(seed))
        non_edges = [tuple(choice(G.vcount(),size=2,replace=False)) for i in range(3*size)]
        non_edges = [x for x in non_edges if G.get_eid(x[0],x[1],directed=True,error=False) == -1]
        non_edges = list(set(non_edges))[:size]
        Gf = ig.Graph.Read_Edgelist(fullgraph,directed=True)
        rem_edges = pd.read_csv(f"{datadir_graphs}/{graph}_s{seed}_rem_edges", sep=' ', header=None)
        rem_edges = np.array(rem_edges)
        np.random.seed(int(seed))
        e = [tuple(choice(Gf.vcount(),size=2,replace=False)) for i in range(int(3*size*0.05))]
        test_e = [x for x in e if Gf.get_eid(x[0],x[1],directed=False,error=False) == -1]
        test_e = list(set(test_e))
        for emb in ["hope","n2v"]:
            with open(f"{data_dir}/link_prediction_{seed}_{emb}_{graph}.csv","w") as fcsv:
                fcsv.write(header)
                for dim in range(2,33,2):
                    for param in ["ppr","katz","aa"]:
                        if emb == "n2v":
                            param = param_trans[param]
                            N2V=True
                            emb_file = f"{datadir_embeddings}/{emb}-{graph}-p{str(param)}-{dim}-{seed}_reduced"
                        elif emb == "hope":
                            emb_file = f"{datadir_embeddings}/{emb}-{graph}-{param}-{dim}-{seed}_reduced"
                            N2V=False
                        start = time()
                        try:
                            out = run_dir_CGE(gr, cl, emb_file)
                            out = eval(out.decode("utf-8"))
                        except:
                            out = [0]*7
                        E = readEmbedding(emb_file,N2V)
                        F = []
                        F_degrees = []
                        for i in G.es:
                            src, dst = i.tuple
                            src_e = np.array(E.loc[src])
                            dst_e = np.array(E.loc[dst])
                            F.append(np.concatenate((src_e, dst_e), axis=0))
                            in_src = G.vs[src].degree(mode="in")
                            out_src = G.vs[src].degree(mode="out")
                            in_dst = G.vs[dst].degree(mode="in")
                            out_dst = G.vs[dst].degree(mode="out")
                            F_degrees.append(np.concatenate((src_e, dst_e, [in_src,out_src,in_dst,out_dst]), axis=0))
                        size = G.ecount()
                        f = [1]*size
                        for i in non_edges:
                            src, dst = i
                            src_e = np.array(E.loc[src])
                            dst_e = np.array(E.loc[dst])
                            F.append(np.concatenate((src_e, dst_e), axis=0))
                            in_src = G.vs[src].degree(mode="in")
                            out_src = G.vs[src].degree(mode="out")
                            in_dst = G.vs[dst].degree(mode="in")
                            out_dst = G.vs[dst].degree(mode="out")
                            F_degrees.append(np.concatenate((src_e, dst_e, [in_src,out_src,in_dst,out_dst]), axis=0))
                        F = np.array(F)
                        F_degrees = np.array(F_degrees)
                        f.extend([0]*size)

                        ## prepare test set, first with all dropped edges from G
                        X_test = []
                        X_test_deg = []
                        X_test_swap = []
                        X_test_swap_deg = []
                        y_test_swap = []
                        for i in rem_edges:
                            src, dst = i
                            in_src = Gf.vs[src].degree(mode="in")
                            out_src = Gf.vs[src].degree(mode="out")
                            in_dst = Gf.vs[dst].degree(mode="in")
                            out_dst = Gf.vs[dst].degree(mode="out")
                            try:
                                src_e = np.array(E.loc[src])
                                dst_e = np.array(E.loc[dst])
                                X_test.append(np.concatenate((src_e, dst_e), axis=0))
                                X_test_deg.append(np.concatenate((src_e, dst_e, [in_src,out_src,in_dst,out_dst]), axis=0))
                                X_test_swap.append(np.concatenate((dst_e, src_e), axis=0))
                                X_test_swap_deg.append(np.concatenate((dst_e, src_e, [in_dst,out_dst,in_src,out_src]), axis=0))
                                y_test_swap.append(0 if Gf.get_eid(dst,src,directed=True,error=False) == -1 else 1)
                            except KeyError:
                                pass
                        size_test = len(X_test)
                        X_test = np.array(X_test)
                        X_test_deg = np.array(X_test_deg)
                        X_test_swap = np.array(X_test_swap)
                        X_test_swap_deg = np.array(X_test_swap_deg)
                        y_test = [1]*size_test

                        ## then for equal number of non-edges (we over-sample to drop edges and collisions from the list)
                        non_edges_test = test_e[:size_test]
                        X_test_nonedge = []
                        X_test_nonedge_deg = []
                        for i in non_edges_test:
                            src, dst = i
                            in_src = Gf.vs[src].degree(mode="in")
                            out_src = Gf.vs[src].degree(mode="out")
                            in_dst = Gf.vs[dst].degree(mode="in")
                            out_dst = Gf.vs[dst].degree(mode="out")
                            try:
                                src_e = np.array(E.loc[src])
                                dst_e = np.array(E.loc[dst])
                                X_test_nonedge.append(np.concatenate((src_e, dst_e), axis=0))
                                X_test_nonedge_deg.append(np.concatenate((src_e, dst_e, [in_src,out_src,in_dst,out_dst]), axis=0))
                            except KeyError:
                                pass
                        X_test_nonedge = np.array(X_test_nonedge)
                        X_test_nonedge_deg = np.array(X_test_nonedge_deg)
                        y_test_nonedge = [0]*size_test

                        # Train XGBoost Classifier
                        model = xgb.XGBClassifier(random_state=seed)
                        model.fit(F, f)

                        pred = model.predict_proba(X_test)[:,1]
                        pred_swap = model.predict_proba(X_test_swap)[:,1]
                        pred_nonedge = model.predict_proba(X_test_nonedge)[:,1]
                        auc = roc_auc_score(y_test+y_test_nonedge, np.concatenate((pred, pred_nonedge), axis=0))
                        auc_swap = roc_auc_score(y_test_swap+y_test_nonedge, np.concatenate((pred_swap, pred_nonedge), axis=0))

                        acc = accuracy_score(y_test+y_test_nonedge, np.concatenate((pred>0.5, pred_nonedge>0.5), axis=0))
                        acc_swap = accuracy_score(y_test_swap+y_test_nonedge, np.concatenate((pred_swap>0.5, pred_nonedge>0.5), axis=0))

                        abs_res = np.abs(pred-pred_swap)
                        diff_score = np.mean(abs_res)
                        std_score = np.std(abs_res)

                        # Train XGBoost Classifier with additional columns
                        model = xgb.XGBClassifier(random_state=seed)
                        model.fit(F_degrees, f)

                        pred = model.predict_proba(X_test_deg)[:,1]
                        pred_swap = model.predict_proba(X_test_swap_deg)[:,1]
                        pred_nonedge = model.predict_proba(X_test_nonedge_deg)[:,1]

                        auc_deg = roc_auc_score(y_test+y_test_nonedge, np.concatenate((pred, pred_nonedge), axis=0))
                        auc_swap_deg = roc_auc_score(y_test_swap+y_test_nonedge, np.concatenate((pred_swap, pred_nonedge), axis=0))

                        acc_deg = accuracy_score(y_test+y_test_nonedge, np.concatenate((pred>0.5, pred_nonedge>0.5), axis=0))
                        acc_swap_deg = accuracy_score(y_test_swap+y_test_nonedge, np.concatenate((pred_swap>0.5, pred_nonedge>0.5), axis=0))

                        abs_res = np.abs(pred-pred_swap)
                        diff_score_deg = np.mean(abs_res)
                        std_score_deg = np.std(abs_res)

                        exec_time = round(time()-start,2)
                        line = f"{graph},{emb},{dim},{str(param)},{exec_time},-1,{out[0]},{out[1]},{out[2]},{out[3]},{out[4]},{out[5]},{out[6]},{seed},{auc},{auc_swap},{acc},{acc_swap},{diff_score},{std_score},{auc_deg},{auc_swap_deg},{acc_deg},{acc_swap_deg},{diff_score_deg},{std_score_deg}\n"
                        fcsv.write(line)
                        print(line)
