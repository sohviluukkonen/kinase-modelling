#General modules
import tqdm
import pandas as pd
import numpy as np

from pulp import *
from typing import List
from collections import defaultdict

#Chemistry modules
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.SimDivFilters import rdSimDivPickers

#ML modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances


#Function for assigning datapoints to clusters for the scaffold-based split
def assignPointsToClusters(picks,fps):
    """
    Assigns points to clusters based on the Tanimoto similarity between the points
    and the cluster centers.
    
    Parameters
    ----------
    picks : list
        The indices of the cluster centers
    fps : list
        The fingerprints of the points to be clustered

    Returns
    -------
    clusters : dict
        A dictionary mapping cluster indices to lists of point indices
    
    Notes
    -----
    This is a simple greedy algorithm that assigns points to the cluster with the
    highest Tanimoto similarity.  It is not guaranteed to find the global optimum.
    """

    clusters = defaultdict(list)
    # add the cluster centers to the clusters
    for i,idx in enumerate(picks):
        clusters[i].append(idx)
    
    # calculate the Tanimoto similarities between the cluster centers and the
    # other points
    sims = np.zeros((len(picks),len(fps)))
    for i in range(len(picks)):
        pick = picks[i]
        sims[i,:] = DataStructs.BulkTanimotoSimilarity(fps[pick],fps)
        sims[i,i] = 0
    
    # assign the points to clusters
    best = np.argmax(sims,axis=0)
    for i,idx in enumerate(best):
        if i not in picks:
            clusters[idx].append(i)
    
    return clusters

#Function for allocating the different clusters to groups for balanced dataset creation (https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/624558fa804882433dffcb67/original/construction-of-balanced-chemically-dissimilar-training-validation-and-test-sets-for-machine-learning-on-molecular-datasets.pdf)
def balance_data_from_tasks_vs_clusters_array_pulp(tasks_vs_clusters_array,
                                              sizes = [1],
                                              equal_weight_perc_compounds_as_tasks = False,
                                              relative_gap = 0,
                                              time_limit_seconds = 60 * 60,
                                              max_N_threads = 1):
    """Linear programming function needed to balance the data while merging clusters

    Parameters
    ----------
    tasks_vs_clusters_array : 2D np.array
        - the cross-tabulation of the number of data points per cluster, per task.
        - columns represent unique clusters.
        - rows represent tasks, except the first row, which represents the number of records (or compounds).
        - Optionally, instead of the number of data points, the provided array may contain the *percentages*
            of data points _for the task across all clusters_ (i.e. each *row*, NOT column, may sum to 1).
        IMPORTANT: make sure the array has 2 dimensions, even if only balancing the number of data records,
            so there is only 1 row. This can be achieved by setting ndmin = 2 in the np.array function.
    sizes : list
        - list of the desired final sizes (will be normalised to fractions internally).
    equal_weight_perc_compounds_as_tasks : bool
        - if True, matching the % records will have the same weight as matching the % data of individual tasks.
        - if False, matching the % records will have a weight X times larger than the X tasks.
    relative_gap : float
        - the relative gap between the absolute optimal objective and the current one at which the solver
          stops and returns a solution. Can be very useful for cases where the exact solution requires
          far too long to be found to be of any practical use.
        - set to 0 to obtain the absolute optimal solution (if reached within the time_limit_seconds)
    time_limit_seconds : int
        - the time limit in seconds for the solver (by default set to 1 hour)
        - after this time, whatever solution is available is returned
    max_N_threads : int
        - the maximal number of threads to be used by the solver.
        - it is advisable to set this number as high as allowed by the available resources.
    
    Output
    ------
    List (of length equal to the number of columns of tasks_vs_clusters_array) of final cluster identifiers
        (integers, numbered from 1 to len(sizes)), mapping each unique initial cluster to its final cluster.
    Example: if sizes == [20, 10, 70], the output will be a list like [3, 3, 1, 2, 1, 3...], where
        '1' represents the final cluster of relative size 20, '2' the one of relative size 10, and '3' the 
        one of relative size 70.
    """
    # Calculate the fractions from sizes

    fractional_sizes = sizes / np.sum(sizes)

    S = len(sizes)

    # Normalise the data matrix
    tasks_vs_clusters_array = tasks_vs_clusters_array / tasks_vs_clusters_array.sum(axis = 1, keepdims = True)

    # Find the number of tasks + compounds (M) and the number of initial clusters (N)
    M, N = tasks_vs_clusters_array.shape
    if (S > N):
        errormessage = 'The requested number of new clusters to make ('+ str(S) + ') cannot be larger than the initial number of clusters (' + str(N) + '). Please review.'
        raise ValueError(errormessage)

    # Given matrix A (M x N) of fraction of data per cluster, assign each cluster to one of S final ML subsets,
    # so that the fraction of data per ML subset is closest to the corresponding fraction_size.
    # The weights on each ML subset (WML, S x 1) are calculated from fractional_sizes harmonic-mean-like.
    # The weights on each task (WT, M x 1) are calculated as requested by the user.
    # In the end: argmin SUM(ABS((A.X-T).WML).WT)
    # where X is the (N x S) binary solution matrix
    # where T is the (M x S) matrix of target fraction sizes (repeat of fractional_sizes)
    # constraint: assign one cluster to one and only one final ML subset
    # i.e. each row of X must sum to 1

    A = np.copy(tasks_vs_clusters_array)

    # Create WT = obj_weights
    if ((M > 1) & (equal_weight_perc_compounds_as_tasks == False)):
        obj_weights = np.array([M-1] + [1] * (M-1))
    else:
        obj_weights = np.array([1] * M)

    obj_weights = obj_weights / np.sum(obj_weights)

    # Create WML
    sk_harmonic = (1 / fractional_sizes) / np.sum(1 / fractional_sizes)

    # Create the pulp model
    prob = LpProblem("Data_balancing", LpMinimize)

    # Create the pulp variables
    # x_names represent clusters, ML_subsets, and are binary variables
    x_names = ['x_'+str(i) for i in range(N * S)]
    x = [LpVariable(x_names[i], lowBound = 0, upBound = 1, cat = 'Integer') for i in range(N * S)]
    # X_names represent tasks, ML_subsets, and are continuous positive variables
    X_names = ['X_'+str(i) for i in range(M * S)]
    X = [LpVariable(X_names[i], lowBound = 0, cat = 'Continuous') for i in range(M * S)]

    # Add the objective to the model

    obj = []
    coeff = []
    for m in range(S):
        for t in range(M):
            obj.append(X[m*M+t])
            coeff.append(sk_harmonic[m] * obj_weights[t])

    prob += LpAffineExpression([(obj[i],coeff[i]) for i in range(len(obj)) ])

    # Add the constraints to the model

    # Constraints forcing each cluster to be in one and only one ML_subset
    for c in range(N):
        prob += LpAffineExpression([(x[c+m*N],+1) for m in range(S)]) == 1

    # Constraints related to the ABS values handling, part 1 and 2
    for m in range(S):
        for t in range(M):
            cs = [c for c in range(N) if A[t,c] != 0]
            prob += LpAffineExpression([(x[c+m*N],A[t,c]) for c in cs]) - X[t] <= fractional_sizes[m]
            prob += LpAffineExpression([(x[c+m*N],A[t,c]) for c in cs]) + X[t] >= fractional_sizes[m]

    # Solve the model
    prob.solve(PULP_CBC_CMD(gapRel = relative_gap, timeLimit = time_limit_seconds, threads = max_N_threads))
    #solver.tmpDir = "/zfsdata/data/erik/erik-rp1/pQSAR/scaffoldsplit_trial/tmp"
    #prob.solve(solver)

    # Extract the solution

    list_binary_solution = [value(x[i]) for i in range(N * S)]
    list_initial_cluster_indices = [(list(range(N)) * S)[i] for i,l in enumerate(list_binary_solution) if l == 1]
    list_final_ML_subsets = [(list((1 + np.repeat(range(S), N)).astype('int64')))[i] for i,l in enumerate(list_binary_solution) if l == 1]
    mapping = [x for _, x in sorted(zip(list_initial_cluster_indices, list_final_ML_subsets))]

    return(mapping)

#Function for sorting targets based on amount of data
def order_targets_per_number_of_datapoints(data, targets):
    """
    Sorts the targets based on the amount of datapoints
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data
    targets : list
        List of targets
        
    Returns
    -------
    list
        List of targets sorted based on the amount of datapoints
    """
    n_datapoints_per_target = {}
    for target in targets:
        n_datapoints_per_target[target] = data[target].count()
    n_datapoints_per_target = dict(sorted(n_datapoints_per_target.items(), key=lambda item: item[1], reverse=True))
    
    return list(n_datapoints_per_target.keys())
    
def random_global_equilibrated_random_split(data, targets, seed):
    """
    Random Global Equilibrated Split (RGES) is done by sorting targets from the target 
    with the most data points to those with the least. Then, for each target, a random split was made. 
    If a compound belonged to a different (train, validation, test) set for a different target, 
    its final label was set to the label of that compound for the target lowest on the sorted list. 
    This mechanism was chosen because reassigning labels for targets with larger numbers of compounds 
    has smaller relative effects on the balance. 
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data
    targets : list
        List of targets
    seed : int
        Seed for the random split
    
    Returns
    -------
    pd.DataFrame
        Dataframe containing the data with a column 'Subset' containing the split"""
    
    print('Create RGES split...')

    ordered_targets = order_targets_per_number_of_datapoints(data, targets)
    
    split_data = data.copy()
    split_data['Subset'] = 0

    for target in tqdm.tqdm(ordered_targets, desc='Targets'):
        index = split_data[~split_data[target].isna()].index.tolist()
        train, test = train_test_split(index, test_size=1/10, random_state=seed)
        train, valid = train_test_split(train, test_size=1/9, random_state=seed)
        for i in train : split_data.loc[i, 'Subset'] = 'train'
        for i in valid : split_data.loc[i, 'Subset'] = 'valid'
        for i in test : split_data.loc[i, 'Subset'] = 'test'

    return split_data

def dissimilaritydrive_global_balanced_cluster_split(data, targets, threads=8, sizes = [0.8,0.1,0.1]):

    """
    Dissimilarity-Driven Global Balanced Cluster Split (DGBC) is done using a method developed in 

    Tricarico, G. A.; Hofmans, J.; Lenselink, E. B.; López-Ramos, M.; Dréanic, M.-P.; Stouten, P. F. W. 
    Construction of balanced, chemically dissimilar training, validation and test sets for machine learning 
    on molecular datasets. 2022, ChemRxiv 2022. DOI: https://doi.org/10.26434/chemrxiv-2022-m8l33-v2. 

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data
    targets : list
        List of targets
    threads : int
        Number of threads to use
    sizes : list
        List of floats containing the size of the train, validation and test set. Default is [0.8,0.1,0.1]

    Returns
    -------
    pd.DataFrame
        Dataframe containing the data with a column 'Subset' containing the split
    """

    print('Create DGBC split...')
    
    # Assign compounds to clusters
    split_data = data.copy().reset_index(drop=True)

    # Compute fingerprints
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 3, 2048) for s in split_data['SMILES']]

    # Get the cluster centers
    print("Pick cluster centers with Sayle's algorithm...")
    lead_picker = rdSimDivPickers.LeaderPicker()
    similarity_threshold = 0.736
    centroids_indices = lead_picker.LazyBitVectorPick(fps, len(fps), similarity_threshold)
    clusters = { i: [centroid_idx] for i, centroid_idx in enumerate(centroids_indices) }

    # Calculate the Tanimoto similarities between the cluster centers 
    # and the other points
    print('Calculating Tanimoto similarities between cluster centers and other points...')
    sims = np.zeros((len(centroids_indices),len(fps)))
    for i, centroid_idx in enumerate(centroids_indices):
        sims[i,:] = DataStructs.BulkTanimotoSimilarity(fps[centroid_idx],fps)
        # sims[i,i] = 0

    # Assign the points to clusters
    print('Assigning points to clusters...')
    best_cluster = np.argmax(sims,axis=0) # shape of best_cluster is (len(fps),)
    for i, idx in enumerate(best_cluster):
        if i not in centroids_indices:
            clusters[idx].append(i)
    
    # # Compute the amount of data per target per cluster
    # # Shape of taks_vs_clusters_array is (n_targets+1, n_clusters)
    target_vs_clusters = np.zeros((len(targets)+1, len(clusters)))
    target_vs_clusters[0,:] = [ len(cluster) for cluster in clusters.values() ]

    for i, target in enumerate(targets):
        for j, indices_per_cluster in clusters.items():
            data_per_cluster = split_data.iloc[indices_per_cluster]
            target_vs_clusters[i+1,j] = data_per_cluster[target].dropna().shape[0]

    print(target_vs_clusters.shape, len(clusters))

    # Balance the data per target per cluster using PuLP and return the mapping of clusters to subsets (train, validation, test)
    mapping = balance_data_from_tasks_vs_clusters_array_pulp(target_vs_clusters,
                                            sizes = sizes,
                                            equal_weight_perc_compounds_as_tasks = False,
                                            relative_gap = 0,
                                            time_limit_seconds = 60 * 60,
                                            max_N_threads = threads,
                                            )

    for i, idx in clusters.items(): 
        if mapping[i] == 1 :
            split_data.loc[idx, 'Subset'] = 'train'
        elif mapping[i] == 2 :
            split_data.loc[idx, 'Subset'] = 'valid'
        elif mapping[i] == 3 :
            split_data.loc[idx, 'Subset'] = 'test'
    
    return split_data

def print_balance_metrics(data : pd.DataFrame, targets : List[str]):
    """ 
    Print the balance metrics for the given subsets and targets.
    """
        
    txt = 'Overall balance:'
    for subset in sorted(data['Subset'].unique()):
        n = len(data[data['Subset'] == subset])
        frac = n/ len(data)
        txt += f' {subset}: {n} ({frac:05.2%})'
    print(txt)
    
    for target in targets:
        txt = f'{target} balance:'
        df = data.dropna(subset=[target])
        for subset in sorted(data['Subset'].unique()):
            n = len(df[df['Subset'] == subset])
            frac = n / len(df)
            txt += f' {subset}: {n} ({frac:05.2%})'
        print(txt)
    
def compute_intersubset_Tanimoto_distance(df, n_jobs=1):
    """
    Compute the minimum Tanimoto distance per compound to the compounds in the other subsets.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data with a column 'Subset' containing the split
    smiles_column : str, optional
        Name of the column containing the SMILES, by default 'SMILES'
    
    Returns
    -------
    pd.DataFrame
        Dataframe containing the data with a column 'MinInterSetTd' containing the minimum Tanimoto distance

    """

    df.reset_index(drop=True, inplace=True)

    # Compute Morgan Fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 3, nBits=2048) for s in df.SMILES]

    # Compute pairwise Tanimoto distances between all compounds
    dists = pairwise_distances(np.array(fps), metric='jaccard', n_jobs=n_jobs)

    for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Computing minimum Tanimoto distance'):
        subset = row['Subset']
        other_subset_indices = df[df['Subset'] != subset].index.values
        min_dist = min([dists[idx, i] for i in other_subset_indices])
        df.loc[idx, 'MinInterSetTd'] = min_dist

    # Print average and std  of minimum distances per subset
    txt = 'Average and std of minimum Tanimoto distance per subset:'
    for subset in sorted(df['Subset'].unique()):
        dist = df[df['Subset'] == subset]['MinInterSetTd'].to_numpy()
        txt += f' {subset}: {np.mean(dist):.2f} ({np.std(dist):.2f})'
    print(txt)

    return df
