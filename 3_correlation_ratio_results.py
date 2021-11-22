from time import time
from common import run_dir_CGE

data_dir = "data"
param_trans = {"ppr":1,
"katz":0.11,
"aa":9}
header = "graph,emb,dim,param,exec_time,landmarks,best_alpha,best_div,best_div_ext,best_div_int,best_alpha_auc,best_auc,best_auc_err\n"

for land in range(100,3001,100):
    with open(f"correlation_ratio.csv","w") as fcsv:
        fcsv.write(header)
        for graph in ["sbm10k","lfr10k","nlfr10k"]:
            gr = f"{data_dir}/{graph}.edgelist"
            cl = f"{data_dir}/{graph}.ecg"
            for emb in ["hope","n2v"]:
                for dim in range(2,33,2):
                    for param in ["ppr","katz","aa"]:
                        if emb == "n2v":
                            param = param_trans[param]
                            emb_file = f"{data_dir}/{emb}-{graph}-p{str(param)}-{dim}"
                        else:
                            emb_file = f"{data_dir}/{emb}-{graph}-{param}-{dim}"
                        start = time()
                        out = run_dir_CGE(gr, cl, emb_file, land)
                        out = eval(out.decode("utf-8"))
                        exec_time = round(time()-start,2)
                        line = f"{graph},{emb},{dim},{str(param)},{exec_time},{land},{out[0]},{out[1]},{out[2]},{out[3]},{out[4]},{out[5]},{out[6]}\n"
                        fcsv.write(line)
