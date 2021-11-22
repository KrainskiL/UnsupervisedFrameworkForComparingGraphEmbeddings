import numpy as np
import scipy.sparse.linalg as lg
import subprocess

def run_dir_CGE(edge_file:str, comm_file:str, embed_file:str, landmarks:int=-1):
    if landmarks == -1:
        return subprocess.check_output(f'julia ./CGE_CLI.jl -g {edge_file} -c {comm_file} -e {embed_file} -d --force-exact --seed 42', shell=True)
    else:
        return subprocess.check_output(f'julia ./CGE_CLI.jl -g {edge_file} -c {comm_file} -e {embed_file} -l {landmarks} -d --seed 42', shell=True)

# Hope embedding with various similarity functions
def Hope(g, sim='katz', dim=2, verbose=False, beta=.01, alpha=.5):
    # For undirected graphs, embedding as source and target are identical
    if g.is_directed() == False:
        dim = dim*2
    A = np.array(g.get_adjacency().data)
    beta = beta
    alpha = alpha
    n = g.vcount()
    # Katz
    if sim == 'katz':
        M_g = np.eye(n) - beta * A
        M_l = beta * A
    # Adamic-Adar
    if sim == 'aa':
        M_g = np.eye(n)
        # fix bug 1/x and take log();
        D = np.diag([1/np.log(x) if x > 1 else 0 for x in g.degree()])
        # D = np.diag([1/np.log(max(2,x)) for x in g.degree()])
        M_l = np.dot(np.dot(A, D), A)
        np.fill_diagonal(M_l, 0)
    # Common neighbors
    if sim == 'cn':
        M_g = np.eye(n)
        M_l = np.dot(A, A)
    # presonalized page rank
    if sim == 'ppr':
        P = []
        for i in range(n):
            s = np.sum(A[i])
            if s > 0:
                P.append([x/s for x in A[i]])
            else:
                P.append([1/n for x in A[i]])
        P = np.transpose(np.array(P))  # fix bug - take transpose
        M_g = np.eye(n)-alpha*P
        M_l = (1-alpha)*np.eye(n)
    S = np.dot(np.linalg.inv(M_g), M_l)
    u, s, vt = lg.svds(S, k=dim//2)
    X1 = np.dot(u, np.diag(np.sqrt(s)))
    X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
    X = np.concatenate((X1, X2), axis=1)
    p_d_p_t = np.dot(u, np.dot(np.diag(s), vt))
    eig_err = np.linalg.norm(p_d_p_t - S)
    if verbose:
        print('SVD error (low rank): %f' % eig_err)
    # undirected graphs have identical source and target embeddings
    if g.is_directed() == False:
        d = dim//2
        return X[:, :d]
    else:
        return X

def saveEmbedding(X, g, fn='_embed'):
    with open(fn, 'w') as f:
        for i in range(X.shape[0]):
            f.write(g.vs[i]['name']+' ')
            for j in range(X.shape[1]):
                f.write(str(X[i][j])+' ')
            f.write('\n')
