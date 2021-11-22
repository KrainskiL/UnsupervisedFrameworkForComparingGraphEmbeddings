from time import time
from common import run_dir_CGE

def gather_CGE_results(output_file:str,data_path:str,landmarks_per_graph:dict=None):
    header = "graph,emb,dim,param,exec_time,landmarks,best_alpha,best_div,best_div_ext,best_div_int,best_alpha_auc,best_auc,best_auc_err\n"
    param_trans = {"ppr":1,"katz":0.11,"aa":9}
    with open(output_file,"w") as fcsv:
        fcsv.write(header)
        for graph in ["sbm10k","lfr10k","nlfr10k","email"]:
            landmarks = landmarks_per_graph[graph] if landmarks_per_graph else -1
            gr = f"{data_path}/{graph}.edgelist"
            cl = f"{data_path}/{graph}.ecg"
            for emb in ["hope","n2v"]:
                for dim in range(2,33,2):
                    for param in ["ppr","katz","aa"]:
                        if emb == "n2v":
                            param = param_trans[param]
                            emb_file = f"{data_path}/{emb}-{graph}-p{str(param)}-{dim}"
                        else:
                            emb_file = f"{data_path}/{emb}-{graph}-{param}-{dim}"
                        start = time()
                        out = run_dir_CGE(gr, cl, emb_file, landmarks)
                        out = eval(out.decode("utf-8"))
                        exec_time = round(time()-start,2)
                        line = f"{graph},{emb},{dim},{str(param)},{exec_time},{landmarks},{out[0]},{out[1]},{out[2]},{out[3]},{out[4]},{out[5]},{out[6]}\n"
                        fcsv.write(line)
                        print(line)
data_dir = "data"

gather_CGE_results("exact_scores.csv",data_dir,None)

landmarks_dict = {"sbm10k":5*30,"lfr10k":5*54,"nlfr10k":5*75,"email":5*42}
gather_CGE_results("approximate_scores.csv",data_dir,landmarks_dict)
