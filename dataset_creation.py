from src.utils import mkdirs
from src.data import load_papyrus_data, retrive_kinase_data_from_Papyrus, filter_data, pivot_Papyrus_data
from src.split import random_global_equilibrated_random_split, \
    dissimilaritydrive_global_balanced_cluster_split, compute_intersubset_Tanimoto_distance

SEED = 2022

if __name__ == '__main__':

    """
    Creating kinase large-scale datasets from Papyrus
    
    WARNING : This script takes a long time to run (around 3h) 
    """
    # Load Papyrus data
    mkdirs('data')
    load_papyrus_data(papyrus_path='data')
    
    # Retrieve, filter & pivot data
    all_kinase_data = retrive_kinase_data_from_Papyrus(source_path='data')
    kinase200, kinase1000 = filter_data(all_kinase_data)
    kinase200 = pivot_Papyrus_data(kinase200)
    kinase1000 = pivot_Papyrus_data(kinase1000)

    # Split data
    targets_kinase200 = kinase200.columns[2:].tolist()
    rges_kinase200 = random_global_equilibrated_random_split(kinase200, targets_kinase200, seed=SEED)
    dgbc_kinase200 = dissimilaritydrive_global_balanced_cluster_split(kinase200, targets_kinase200, sizes=[0.8,0.1,0.1] )

    targets_kinase1000 = kinase1000.columns[2:].tolist()
    rges_kinase1000 = random_global_equilibrated_random_split(kinase1000, targets_kinase1000, seed=SEED)
    dgbc_kinase1000 = dissimilaritydrive_global_balanced_cluster_split(kinase1000, targets_kinase1000, sizes=[0.8,0.1,0.1] )

    # Compute Tanimoto distance
    rges_kinase200 = compute_intersubset_Tanimoto_distance(rges_kinase200)
    dgbc_kinase200 = compute_intersubset_Tanimoto_distance(dgbc_kinase200)

    rges_kinase1000 = compute_intersubset_Tanimoto_distance(rges_kinase1000)
    dgbc_kinase1000 = compute_intersubset_Tanimoto_distance(dgbc_kinase1000)

    # Save data
    mkdirs('data/datasets')
    rges_kinase200.to_csv('data/datasets/kinase200_RGES.csv.gz', index=False)
    dgbc_kinase200.to_csv('data/datasets/kinase200_DGBC.csv.gz', index=False)
    rges_kinase1000.to_csv('data/datasets/kinase1000_RGES.csv.gz', index=False)
    dgbc_kinase1000.to_csv('data/datasets/kinase1000_DGBC.csv.gz', index=False)







