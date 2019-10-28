import numpy as np
from scipy.optimize import linear_sum_assignment

def row_param_cost(global_atoms, atoms_j_l, global_inv_sigmas, sigma_inv_j):
    
    match_norms = ((atoms_j_l + global_atoms)**2/(sigma_inv_j + global_inv_sigmas)).sum(axis=1) - (global_atoms**2/global_inv_sigmas).sum(axis=1)
    
    return match_norms
    
def compute_cost(global_atoms, atoms_j, global_inv_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma, popularity_counts, gamma, J):
    
    Lj = atoms_j.shape[0]
    counts = np.minimum(np.array(popularity_counts),10)
#    counts = np.array(popularity_counts)
    param_cost = np.array([row_param_cost(global_atoms, atoms_j[l], global_inv_sigmas, sigma_inv_j) for l in range(Lj)])
    param_cost += 2*np.log(counts/(J-counts))
    
    ## Nonparametric cost
    L = global_atoms.shape[0]
    max_added = min(Lj, max(700 - L, 1))
#    max_added = Lj
    nonparam_cost = np.outer((((atoms_j + prior_mean_norm)**2/(prior_inv_sigma+sigma_inv_j)).sum(axis=1) - (prior_mean_norm**2/prior_inv_sigma).sum()),np.ones(max_added))
    cost_pois = 2*np.log(np.arange(1,max_added+1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2*np.log(gamma/J)
    
    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost

def matching_upd_j(atoms_j, global_atoms, sigma_inv_j, global_inv_sigmas, prior_mean_norm, prior_inv_sigma, popularity_counts, gamma, J):
    
    L = global_atoms.shape[0]
        
    full_cost = compute_cost(global_atoms, atoms_j, global_inv_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma, popularity_counts, gamma, J)
    

    row_ind, col_ind = linear_sum_assignment(-full_cost)

    assignment_j = []
    
    new_L = L
    
    for l, i in zip(row_ind, col_ind):
        if i < L:
            popularity_counts[i] += 1
            assignment_j.append(i)
            global_atoms[i] += atoms_j[l]
            global_inv_sigmas[i] += sigma_inv_j
        else: # new neuron
            popularity_counts += [1]
            assignment_j.append(new_L)
            new_L += 1
            global_atoms = np.vstack((global_atoms,prior_mean_norm + atoms_j[l]))
            global_inv_sigmas = np.vstack((global_inv_sigmas,prior_inv_sigma + sigma_inv_j))

    return global_atoms, global_inv_sigmas, popularity_counts, assignment_j

def objective(global_weights, global_inv_sigmas):
    obj = ((global_weights)**2/global_inv_sigmas).sum()
    return obj
    
def match_local_atoms(local_atoms, sigma, sigma0, gamma, it, mean_prior=0):
    J = len(local_atoms)
    D = local_atoms[0].shape[1]
    
    group_order = sorted(range(J), key = lambda x: -local_atoms[x].shape[0])
    
    sigma_inv_local = [np.ones(D)/sigma for _ in range(J)]
    sigma_inv_prior = np.ones(D)/sigma0
    mean_prior = np.ones(D)*mean_prior
    
    local_atoms_norm = [w*s for w,s in zip(local_atoms,sigma_inv_local)]
    prior_mean_norm = mean_prior*sigma_inv_prior
    
    global_atoms = prior_mean_norm + local_atoms_norm[group_order[0]]
    global_inv_sigmas = np.outer(np.ones(global_atoms.shape[0]),sigma_inv_prior + sigma_inv_local[group_order[0]])
    
    popularity_counts = [1]*global_atoms.shape[0]
    
    assignment = [[] for _ in range(J)]
    
    assignment[group_order[0]] = list(range(global_atoms.shape[0]))
    
    ## Initialize
    for j in group_order[1:]:
        global_atoms, global_inv_sigmas, popularity_counts, assignment_j = matching_upd_j(local_atoms_norm[j], global_atoms, sigma_inv_local[j], global_inv_sigmas, prior_mean_norm, sigma_inv_prior, popularity_counts, gamma, J)
        assignment[j] = assignment_j
    
    ## Iterate over groups
    for iteration in range(it):
        random_order = np.random.permutation(J)
        for j in random_order: #random_order:
            to_delete = []
            ## Remove j
            Lj = len(assignment[j])
            for l, i in sorted(zip(range(Lj),assignment[j]), key = lambda x: -x[1]):
                popularity_counts[i] -= 1
                if popularity_counts[i] == 0:
                    del popularity_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, l_ind in enumerate(assignment[j_clean]):
                            if i < l_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                            elif i == l_ind and j_clean != j:
                                print('Warning - weird unmatching')
                else:
                    global_atoms[i] = global_atoms[i] - local_atoms_norm[j][l]
                    global_inv_sigmas[i] -= sigma_inv_local[j]
                    
            global_atoms = np.delete(global_atoms,to_delete,axis=0)
            global_inv_sigmas = np.delete(global_inv_sigmas,to_delete,axis=0)
    
            ## Match j
            global_atoms, global_inv_sigmas, popularity_counts, assignment_j = matching_upd_j(local_atoms_norm[j], global_atoms, sigma_inv_local[j], global_inv_sigmas, prior_mean_norm, sigma_inv_prior, popularity_counts, gamma, J)
            assignment[j] = assignment_j
        
        print('Matching iteration %d' % iteration)
        print('Objective (without prior) at iteration %d is %f; number of global atoms is %d' % (iteration,objective(global_atoms, global_inv_sigmas),global_atoms.shape[0]))
        
    print('Number of global atoms is %d, gamma %f' % (global_atoms.shape[0], gamma))
    
    map_out = np.array([g_a/g_s for g_a,g_s in zip(global_atoms,global_inv_sigmas)])
    return map_out, popularity_counts
#    return assignment, global_atoms, global_sigmas