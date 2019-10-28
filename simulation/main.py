from data_sampling import gen_partition, hungarian_match, min_match
from gaus_marginal_matching import match_local_atoms
import numpy as np
import pickle, copy
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Data generating parameters
L = 50 # number of global atoms
Js = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100] # number of groups
D = 50 # data dimension
M = 5000 # data points per group (not relevant if we use true local atoms)
local_sd_arr = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
global_sd = np.sqrt(L) # sigma0
mu0 = 2. # mu0
trials = 10

res_J = {}


for J in Js:
        print('#'*100)
        print('Running for J = %d' % J)
        print('#' * 100)
        
        res_J[J] = {}
        res = {}
        
        for local_sd in local_sd_arr:

                print('-'*100)
                print('Local sd: ', local_sd)

                res[local_sd] = {
                        'matching_true': [],
                        'kmeans_matching': [],
                        'kmeans_on_kmeans': [],
                        'pooled_kmeans': [],
                        'matching_true_params': {
                                'mu0': [], 'sigma': [], 'sigma0': [], 'L': [], 'L_true': []
                        },
                        'kmeans_matching_params': {
                                'mu0': [], 'sigma': [], 'sigma0': [], 'L': [], 'L_true': []
                        }
                }

                for trial in range(trials):
                        print("Executing trial ", trial)
                        ## Generate some data
                        np.random.seed(trial)
                        global_atoms, global_p, data, atoms = gen_partition(L=L, J=J, D=D, M=M, local_sd=local_sd, mu0=mu0, global_sd=global_sd)

                        ## Run matching
                        ## Input:
                        # local_atoms - list of len J of numpy arrays of size Lj x D, where Lj is number of local atoms
                        # sigma and sigma0 - initial guesses for hyperparameters
                        # gamma - higgher values encourage more global atoms (this parameter is not estimated)
                        # it - number of Hungarian algorithm passes through all groups
                        # optimize_hyper - enable hyperparameter estimation
                        est_atoms, popularity_counts, (mu0_est, sigma, sigma0) = match_local_atoms(local_atoms=atoms, sigma=1., sigma0=1., gamma=1., it=10, optimize_hyper=True)
                        print(est_atoms.shape)

                        ## Quality of estimated global atoms
                        obj_by_component_true = hungarian_match(est_atoms, global_atoms)
                        print('Using true local atoms')
                        print('Total matching error is %f' % sum(obj_by_component_true)) # this only makes sense when estimated number of global atoms is correct
                        print('Minimum matching error is %f' % min_match(est_atoms, global_atoms)) # this is more reliable metric

                        res[local_sd]['matching_true'].append(min_match(est_atoms, global_atoms))
                        res[local_sd]['matching_true_params']['mu0'].append(mu0_est)
                        res[local_sd]['matching_true_params']['sigma'].append(sigma)
                        res[local_sd]['matching_true_params']['sigma0'].append(sigma0)
                        res[local_sd]['matching_true_params']['L'].append(est_atoms.shape[0])
                        res[local_sd]['matching_true_params']['L_true'].append(global_atoms.shape[0])

                        ## Estimating local atoms from data using kmeans for each group
                        from sklearn.cluster import KMeans
                        atoms_est = []
                        for j in range(J):
                            atoms_j_est = KMeans(n_clusters=atoms[j].shape[0], n_init=8, n_jobs=8).fit(data[j]).cluster_centers_
                            atoms_est.append(atoms_j_est)

                        est_atoms_kmeans, popularity_counts_kmeans, (mu0_est, sigma, sigma0) = match_local_atoms(local_atoms=atoms_est, sigma=1., sigma0=1., gamma=1., it=10)
                        obj_by_component = hungarian_match(est_atoms_kmeans, global_atoms)
                        print('\nUsing estimated local atoms')
                        print('Total matching error is %f' % sum(obj_by_component))
                        print('Minimum matching error is %f' % min_match(est_atoms_kmeans, global_atoms))

                        res[local_sd]['kmeans_matching'].append(min_match(est_atoms_kmeans, global_atoms))
                        res[local_sd]['kmeans_matching_params']['mu0'].append(mu0_est)
                        res[local_sd]['kmeans_matching_params']['sigma'].append(sigma)
                        res[local_sd]['kmeans_matching_params']['sigma0'].append(sigma0)
                        res[local_sd]['kmeans_matching_params']['L'].append(est_atoms_kmeans.shape[0])
                        res[local_sd]['kmeans_matching_params']['L_true'].append(global_atoms.shape[0])

                        ## Baseline: KMeans on KMeans
                        atoms_est_pooled = np.vstack(atoms_est)
                        kmeans_est_atoms_kmeans = KMeans(n_clusters=L, n_init=8, n_jobs=8, max_iter=1000).fit(atoms_est_pooled).cluster_centers_
                        min_matching_err_kmeans2 = min_match(kmeans_est_atoms_kmeans, global_atoms)
                        print('\nKMeans on KMeans estimated atoms')
                        print('Minimum matching error is %f' % min_matching_err_kmeans2)
                        res[local_sd]['kmeans_on_kmeans'].append(min_matching_err_kmeans2)

                        ## Baseline: kmeans on aggregated dataset from all groups
                        data_pooled = np.vstack(data)
                        pooled_kmeans_atoms = KMeans(n_clusters=global_atoms.shape[0], n_init=8, n_jobs=8, max_iter=1000).fit(data_pooled).cluster_centers_
                        obj_by_component_pooled = hungarian_match(pooled_kmeans_atoms, global_atoms)
                        print('\nUsing pooled data K-means')
                        print('Total matching error is %f' % sum(obj_by_component_pooled))
                        print('Minimum matching error is %f' % min_match(pooled_kmeans_atoms, global_atoms))

                        res[local_sd]['pooled_kmeans'].append(min_match(pooled_kmeans_atoms, global_atoms))

                        print('-'*100)

                print("****" * 100)
                print(res)

        res_J[J] = copy.deepcopy(res)

        with open('results-J.p', 'wb') as f:
                pickle.dump(res_J, f)


# Plot data

style = {"linewidth":3, "markeredgewidth":4, "elinewidth":2, "capsize":5, "linestyle":'--'}

def plot_mmerr_J(res):
    true_Js = sorted(res.keys())
    
    prop_dict = {'true':['green',(2,5),'v', r'SPAHM (true local atoms)'] , 
                 'kmeans':['red',(5,2),'o','SPAHM (estimated local atoms)'], 
                 'pooled':['orange',(5,2,20,2),'*','K-Means (Pooled)'],
                 'kmeans_on_kmeans': ['blue', (4,10), 's', 'K-Means "matching"']
                }
    noise_scale = 0.7
    sig_exp = 1.0
    
    matching_true_mu, matching_true_std = [], []
    matching_kmeans_mu, matching_kmeans_std = [], []
    pooled_kmeans_mu, pooled_kmeans_std = [], []
    kmeans_on_kmeans_mu, kmeans_on_kmeans_std = [], []
    
    for J in true_Js:
        matching_true_mu.append(np.mean(res[J][sig_exp]['matching_true']))
        matching_true_std.append(np.std(res[J][sig_exp]['matching_true']))
        
        matching_kmeans_mu.append(np.mean(res[J][sig_exp]['kmeans_matching']))
        matching_kmeans_std.append(np.std(res[J][sig_exp]['kmeans_matching']))
        
        pooled_kmeans_mu.append(np.mean(res[J][sig_exp]['pooled_kmeans']))
        pooled_kmeans_std.append(np.std(res[J][sig_exp]['pooled_kmeans']))
        
        kmeans_on_kmeans_mu.append(np.mean(res[J][sig_exp]['kmeans_on_kmeans']))
        kmeans_on_kmeans_std.append(np.std(res[J][sig_exp]['kmeans_on_kmeans']))
        
    plt.figure(figsize=(12, 12))
    
    for (k, (y, yerr)) in zip(['true', 'kmeans', 'pooled', 'kmeans_on_kmeans'], [(matching_true_mu, matching_true_std), (matching_kmeans_mu, matching_true_std), (pooled_kmeans_mu, pooled_kmeans_std), (kmeans_on_kmeans_mu, kmeans_on_kmeans_std)]):
        color, dashes, marker, label = prop_dict[k]
        error = np.random.uniform(-noise_scale, +noise_scale, len(true_Js))
        plt.errorbar(x=true_Js + error, y=y, yerr=yerr, label=label, color=color, dashes=dashes, marker=marker, **style)
    
    plt.xlabel('J', fontsize=26)
    plt.ylabel('Housdorff Distance', fontsize=26)
    plt.legend(frameon=True, fontsize=22)
    plt.tick_params(labelsize=22)
    plt.grid()
    plt.savefig('mmerr-J.pdf', bbox_inches='tight')


def plot_mmerr_std(res):
    true_sigmas = sorted(res.keys())
    
    prop_dict = {'true':['green',(2,5),'v', r'SPAHM (true local atoms)'] , 
                 'kmeans':['red',(5,2),'o','SPAHM (estimated local atoms)'], 
                 'pooled':['orange',(5,2,20,2),'*','K-Means (Pooled)'],
                 'kmeans_on_kmeans': ['blue', (4,10), 's', 'K-Means "matching"']
                }
    noise_scale = 0.1
    
    matching_true_mu, matching_true_std = [], []
    matching_kmeans_mu, matching_kmeans_std = [], []
    pooled_kmeans_mu, pooled_kmeans_std = [], []
    kmeans_on_kmeans_mu, kmeans_on_kmeans_std = [], []
    
    for sig in true_sigmas:
        matching_true_mu.append(np.mean(res[sig]['matching_true']))
        matching_true_std.append(np.std(res[sig]['matching_true']))
        
        matching_kmeans_mu.append(np.mean(res[sig]['kmeans_matching']))
        matching_kmeans_std.append(np.std(res[sig]['kmeans_matching']))
        
        pooled_kmeans_mu.append(np.mean(res[sig]['pooled_kmeans']))
        pooled_kmeans_std.append(np.std(res[sig]['pooled_kmeans']))
        
        kmeans_on_kmeans_mu.append(np.mean(res[sig]['kmeans_on_kmeans']))
        kmeans_on_kmeans_std.append(np.std(res[sig]['kmeans_on_kmeans']))
        
    plt.figure(figsize=(12, 12))
    
    for (k, (y, yerr)) in zip(['true', 'kmeans', 'pooled', 'kmeans_on_kmeans'], [(matching_true_mu, matching_true_std), (matching_kmeans_mu, matching_true_std), (pooled_kmeans_mu, pooled_kmeans_std), (kmeans_on_kmeans_mu, kmeans_on_kmeans_std)]):
        color, dashes, marker, label = prop_dict[k]
        error = np.random.uniform(-noise_scale, +noise_scale, len(true_sigmas))
        plt.errorbar(x=true_sigmas + error, y=y, yerr=yerr, label=label, color=color, dashes=dashes, marker=marker, **style)

    plt.xlabel('$\sigma$', fontsize=26)
    plt.ylabel('Hausdorff Distance', fontsize=26)
    plt.legend(frameon=True, fontsize=22)
    plt.tick_params(labelsize=22)
    plt.grid()
    plt.savefig('mmerr.pdf', bbox_inches='tight')


plot_mmerr_J(res_J)
plot_mmerr_std(res_J[20])

