from time import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as PRF, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARS
from sklearn.cluster import KMeans

def readEmbedding(fn="_embed", N2V=False):
    if N2V:
        D = pd.read_csv(fn, sep=' ', header=None, skiprows=1)
        D = D.sort_values(by=0)
    else:
        D = pd.read_csv(fn, sep=' ', header=None)
    return np.array(D.iloc[:,1:])

def run_task(output_file:str,data_path:str,iterations:int,task:str):
    param_trans = {"ppr":1,"katz":0.11,"aa":9}
    clusters = {"sbm10k":30,"lfr10k":54,"nlfr10k":75,"email":42}
    if task == "comm_detect":
            header = "graph,emb,dim,param,exec_time,landmarks,model,iter,accuracy,precision,recall,fscore\n"
    elif task == "clustering":
        header = "graph,emb,dim,param,exec_time,landmarks,model,clusters,iter,ami,apr\n"

    with open(output_file,"w") as fcsv:
        fcsv.write(header)
        for graph in ["sbm10k","lfr10k","nlfr10k","email"]:
            gr = f"{data_path}/{graph}.edgelist"
            cl = f"{data_path}/{graph}.ecg"
            for emb in ["hope","n2v"]:
                for dim in range(2,33,2):
                    for param in ["ppr","katz","aa"]:
                        if emb == "n2v":
                            param = param_trans[param]
                            emb_file = f"{data_path}/{emb}-{graph}-p{str(param)}-{dim}"
                            N2V = True
                        else:
                            emb_file = f"{data_path}/{emb}-{graph}-{param}-{dim}"
                            N2V = False
                        for seed in range(iterations):
                            start = time()
                            X = readEmbedding(emb_file,N2V)
                            X = np.array([r[~np.isnan(r)] for r in X])
                            y = np.loadtxt(cl,dtype='uint16',usecols=(0))
                            if task == "comm_detect":
                                # Split dataset with 75/25 ratio and fixed seed
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
                                # Train XGBoost Classifier
                                model = xgb.XGBClassifier(objective='multi:softmax', random_state=seed)
                                model.fit(X_train, y_train)
                                #Metrics
                                pred = model.predict(X_test)
                                acc = sum(y_test == pred)/y_test.shape[0]
                                prec, rec, fscore, _ = PRF(y_test, pred,average="macro")
                                exec_time = round(time()-start,2)
                                line = f"{graph},{emb},{dim},{str(param)},{exec_time},-1,xgboost,{str(seed)},{str(acc)},{str(prec)},{str(rec)},{str(fscore)}\n"
                            elif task == "clustering":
                                # Train kmeans
                                clusts = clusters[graph]
                                kmeans = KMeans(n_clusters=clusts, random_state=seed).fit(X)
                                pred = kmeans.labels_
                                #Metrics
                                ami = AMI(y,pred)
                                ars = ARS(y,pred)
                                exec_time = round(time()-start,2)
                                line = f"{graph},{emb},{dim},{str(param)},{exec_time},-1,kmeans,{str(clusts)},{str(seed)},{str(ami)},{str(ars)}\n"
                            fcsv.write(line)
                            print(line)

data_dir = "data"

run_task("comm_detection.csv",data_dir,10,"comm_detect")
run_task("clustering.csv",data_dir,20,"clustering")
