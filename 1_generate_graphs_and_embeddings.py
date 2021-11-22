import os
import random
import numpy as np
import igraph as ig
import networkx as nx
from node2vec import Node2Vec

from common import *

#Create 10k graphs
datadir = 'data'
os.makedirs(datadir, exist_ok=True)

#SBM
n = 10000
C = 30
p = 0.025
q = 0.001
pm = np.full((C, C), q)
np.fill_diagonal(pm, p)
random.seed(42)
g = ig.Graph.Preference(n=n, type_dist=list(
    np.repeat(1/C, C)), pref_matrix=list(pm), attribute='class', directed=True)
g.vs['name'] = [str(i) for i in np.arange(g.vcount())]

with open(f"{datadir}/sbm10k.ecg", "w") as f:
    for e in g.vs['class']:
        f.write(str(e)+"\n")
with open(f"{datadir}/sbm10k.edgelist", "w") as f:
    for e in g.get_edgelist():
        f.write(str(e[0])+" "+str(e[1])+"\n")

#LFR 10k

# N=10000, k=N/100=100, maxk=N/20=500
# LFR graph withn= 10000, μ=.5 and power law exponents (2,1)
os.system("./lfrbench_udwov - name lfr10k - N 10000 - k 100 - maxk 500 - muw 0.5 - t1 2 - t2 1 - seed seed.txt - a 1")

os.system(f'mv lfr10k* {datadir}')
os.system(f"cut -f2 {datadir}/lfr10k.nmc > {datadir}/lfr10k.ecg")
os.system(f"cut -f 1-2 {datadir}/lfr10k.nsa > {datadir}/lfr10k.edgelist")
fname = f"{datadir}/lfr10k.edgelist"
os.system(f'tail -n +2 "{fname}" > "{fname}.tmp" && mv "{fname}.tmp" "{fname}"')

f = open(f"{datadir}/lfr10k.edgelist").readlines()
f2 = []
for e in f:
    line = e.replace("\n", "").replace("\t", " ").split(" ")
    f2.append([str(int(line[0])-1), str(int(line[1])-1)])
with open(f"{datadir}/lfr10k.edgelist", "w") as f:
    for e in f2:
        f.write(e[0]+" "+e[1]+"\n")

#noisy LFR 10k

# N=10000, k=N/100=100, maxk=N/20=500
# LFR graph withn= 10000,μ=.2 and power law exponents (3,2).
os.system("./lfrbench_udwov - name nlfr10k - N 10000 - k 100 - maxk 500 - muw 0.20 - t1 3 - t2 2 - seed seed.txt - a 1")

os.system(f'mv nlfr10k* {datadir}')
os.system(f"cut -f2 {datadir}/nlfr10k.nmc > {datadir}/nlfr10k.ecg")
os.system(f"cut -f 1-2 {datadir}/nlfr10k.nsa > {datadir}/nlfr10k.edgelist")
fname = f"{datadir}/nlfr10k.edgelist"
os.system(f'tail -n +2 "{fname}" > "{fname}.tmp" && mv "{fname}.tmp" "{fname}"')

f = open(f"{datadir}/nlfr10k.edgelist").readlines()
f2 = []
for e in f:
    line = e.replace("\n", "").replace("\t", " ").split(" ")
    f2.append([str(int(line[0])-1), str(int(line[1])-1)])
with open(f"{datadir}/nlfr10k.edgelist", "w") as f:
    for e in f2:
        f.write(e[0]+" "+e[1]+"\n")

# Generate embeddings

# LOOP over dim/sim
for graph in ["sbm10k", "lfr10k", "nlfr10k","email"]:
    g = ig.Graph.Read_Edgelist(f"{datadir}/{graph}.edgelist", directed=True)
    g.vs['name'] = [str(i) for i in np.arange(g.vcount())]
    for dim in np.arange(2, 33, 2):
        for sim in ['ppr', 'katz', 'aa']:
            X = Hope(g, sim=sim, dim=dim)
            fname = f'{datadir}/hope-{graph}-{sim}-{dim}'
            saveEmbedding(X, g, fn=fname)
            print(fname)

for graph in ["sbm10k", "lfr10k", "nlfr10k","email"]:
    g = nx.read_edgelist(f'{datadir}/{graph}.edgelist',
                         create_using=nx.DiGraph())
    for dim in np.arange(2, 33, 2):
        for p_val in [1/9, 1, 9]:
            node2vec = Node2Vec(g, dimensions=dim, p=p_val,
                                q=1/p_val, quiet=True, seed=42)
            model = node2vec.fit()
            fname = f"{datadir}/n2v-{graph}-p{round(p_val,2)}-{dim}"
            model.wv.save_word2vec_format(fname)
            print(fname)
