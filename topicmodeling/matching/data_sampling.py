import numpy as np
import sys
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment

def gen_gaus_mixture(centers, mixing_prop=None, noise_sd=0.1, M=5000):
#    noise_sd=0.1
#    M=5000
#    mixing_prop=None
#    centers = np.random.normal(0,10,(10,500))
    
    K, D = centers.shape
    if mixing_prop is None:
        mixing_prop = np.random.dirichlet(np.ones(K))
    assignments = np.random.choice(K, size=M, p=mixing_prop)
    data_j = np.zeros((M,D))
    for k in range(K):
        data_j[assignments==k] = np.random.normal(loc=centers[k], scale=noise_sd, size=((assignments==k).sum(),D))
        
    return data_j

def gen_partition(L, J, D, M=5000, base_measure='gaus', ll='gaus', local_model='mixture', local_sd=0.1, global_sd=None, mu0=0, a=1, b=1):
#    base_measure=='gaus'
#    ll = 'gaus'
#    local_model='mixture'
#    noise_local_sd=0.1
#    a=1
#    b=1
#    L = 100
#    J = 10
#    D = 50
#    M = 5000
    
    if global_sd is None:
        global_sd = np.sqrt(L)
        
    global_p = np.random.beta(a=a, b=b, size=L)

    # MANUAL
    idxs_list = [[0, 1], [1,2], [0,2]]
    
    if base_measure=='gaus':
        global_atoms = np.random.normal(mu0,global_sd,size=(L,D))
    else:
        sys.exit('unsupported base measure')
        
    data = []
    used_components = set()
    atoms = []
    for j in range(J):
        atoms_j_idx = [l for l in range(L) if np.random.binomial(1, global_p[l])]
        used_components.update(atoms_j_idx)
        
        if ll=='gaus':
            atoms_j = np.random.normal(global_atoms[atoms_j_idx], scale=local_sd)
        else:
            sys.exit('unsupported likelihood kernel')
            
        if local_model=='mixture':
            data_j = gen_gaus_mixture(atoms_j, M=M)
        else:
            sys.exit('unsupported local model')
            
        data.append(data_j)
        atoms.append(atoms_j)
        
    if len(used_components)<L:
        L = len(used_components)
        L_idx = list(used_components)
        global_p = global_p[L_idx]
        global_atoms = global_atoms[L_idx]
        print('Removing unused components; new L is %d' % L)
        
    return global_atoms, global_p, data, atoms
    
def hungarian_match(est_atoms, true_atoms):
#    true_atoms = global_atoms
#    est_atoms = global_atoms[np.random.permutation(global_atoms.shape[0])]
    
    dist = euclidean_distances(true_atoms, est_atoms)
    
    row_ind, col_ind = linear_sum_assignment(dist)
    
    obj = []
    for r, c in zip(row_ind, col_ind):
        obj.append(dist[r,c])
    return obj

def min_match(est_atoms, true_atoms):
    e_to_t = np.apply_along_axis(lambda x: np.sqrt(((true_atoms-x)**2).sum(axis=1)), 1, est_atoms)
    return max([max(np.min(e_to_t, axis=0)), max(np.min(e_to_t, axis=1))])