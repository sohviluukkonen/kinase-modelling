#General modules
import sys
import tqdm
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
from pulp import *
from multiprocessing import cpu_count

#Chemistry modules
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.SimDivFilters import rdSimDivPickers

#ML modules
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from .utils import compute_fps

def parse_args():
    parsera = argparse.ArgumentParser(description='Run the GLPG model workflow')
    parsera.add_argument("-datafile", help="the datafile with affinity values to load, pivotted format with smiles as index and targets as columns")
    # parsera.add_argument("-splitdir", help = "folder to store splits in")
    parsera.add_argument("-output", help = "output prefix")
    parsera.add_argument("-splittype", help = "choose splittype, options: random, time and scaffold") 
    parsera.add_argument("-nsplits", help = "number of splits")
    parsera.add_argument("-seed", help = "(random split) integer that sets the randomness of the split for reproducibility")
    parsera.add_argument("-threads", help = "(scaffold split) number of threads used for making the scaffold split")
    global args
    args=parsera.parse_args()

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

    ordered_targets = order_targets_per_number_of_datapoints(data, targets)
    
    split_data = data.copy()
    split_data['Subset'] = 0
    
    for target in ordered_targets:
        index = split_data[~split_data[target].isna()].index.tolist()
        train, test = train_test_split(index, test_size=0.1, random_state=seed)
        train, valid = train_test_split(train, test_size=0.11, random_state=seed)
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
    
    # Assign compounds to clusters
    split_data = data.copy()
    PandasTools.AddMoleculeColumnToFrame(split_data, "SMILES", 'Molecule')
    FP = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,3,2048) for mol in split_data['Molecule']]
    lp = rdSimDivPickers.LeaderPicker()
    thresh = 0.736 # <- minimum distance between cluster centroids
    picks = lp.LazyBitVectorPick(FP,len(FP),thresh)
    clusters = assignPointsToClusters(picks,FP)
    cluster_size = [len(j) for i,j in clusters.items()]
    
    # Compute the amount of data per target per cluster
    # Shape of taks_vs_clusters_array is (n_targets+1, n_clusters)
    tasks_vs_clusters_list = [cluster_size] # Add cluster size to the list
    for target in targets:  
        amount_cluster_target = []
        for i, j in clusters.items():
            cluster = data.iloc[j] 
            cluster_target = cluster[target] 
            amount_cluster_target.append(cluster_target.dropna().size) 
        tasks_vs_clusters_list.append(amount_cluster_target)
    tasks_vs_clusters_array = np.array(tasks_vs_clusters_list)

    # Balance the data per target per cluster using PuLP and return the mapping of clusters to subsets (train, validation, test)
    mapping = balance_data_from_tasks_vs_clusters_array_pulp(tasks_vs_clusters_array,
                                            sizes = sizes,
                                            equal_weight_perc_compounds_as_tasks = False,
                                            relative_gap = 0,
                                            time_limit_seconds = 60*30,
                                            max_N_threads = threads)
    grouped_clusters = pd.DataFrame({"group": mapping,
                                    "cluster": [j for i, j in clusters.items()]})
    del split_data["Molecule"]
    
    split_data['Split'] = 'Scaffold'
    split_data["Subset"] = 0
    for i in grouped_clusters["cluster"].loc[grouped_clusters["group"]==1]: split_data.loc[i,['Subset']] = 'train'
    for i in grouped_clusters["cluster"].loc[grouped_clusters["group"]==2]: split_data.loc[i,['Subset']] = 'valid'
    for i in grouped_clusters["cluster"].loc[grouped_clusters["group"]==3]: split_data.loc[i,['Subset']] = 'test'
    
    return split_data
    
def tanimoto_distance_matrix(fp_list):
    """Calculate distance matrix for fingerprint list"""
    dissimilarity_matrix = []
    # Notice how we are deliberately skipping the first and last items in the list
    # because we don't need to compare them against themselves
    for i in tqdm.tqdm(range(1, len(fp_list)), desc='Calculating distance matrix'):
        # Compare the current fingerprint against all the previous ones in the list
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix

def compute_intersubset_Tanimoto_distance(df):
    """
    Compute the minimum Tanimoto distance per compound to the compounds in the other subsets.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data with a column 'Subset' containing the split
    
    Returns
    -------
    pd.DataFrame
        Dataframe containing the data with a column 'MinInterSetTd' containing the minimum Tanimoto distance
    """
    n = len(df)

    # Compute Morgan Fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, nBits=2048) for smiles in tqdm.tqdm(df.SMILES, desc='Computing Morgan Fingerprints from SMILES')]
    
    # Compute Tanimoto distance matrix
    dist_matrix_flatten = tanimoto_distance_matrix(fps)

    # Create symmetric distance matrix
    dist_matrix = np.zeros((n, n))
    dist_matrix[np.triu_indices(n, k=1)] = dist_matrix_flatten
    dist_matrix += dist_matrix.T

    # Compute minimum distance per compound to the compounds in the other subsets
    for j in ['train', 'valid', 'test']:
        ref_idx = df[df.Subset == j].index.tolist()
        other_idx = df[df.Subset != j].index.tolist()
        for i in tqdm.tqdm(ref_idx, total=len(ref_idx), desc='Computing minimum interset Tanimoto distance for mols in subset {}'.format(j)) :
            df.loc[i, 'MinInterSetTd'] = min(dist_matrix[i, other_idx])
    
    return df