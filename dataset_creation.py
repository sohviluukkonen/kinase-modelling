import pandas as pd

from papyrus_scripts.download import download_papyrus

from src.utils import mkdirs
from src.data import retrieve_kinase_data_from_Papyrus, filter_data, pivot_Papyrus_data
from src.split import random_global_equilibrated_random_split, print_balance_metrics, \
    dissimilaritydrive_global_balanced_cluster_split, compute_intersubset_Tanimoto_distance
from src.inputs import create_input_files

SEED = 2022

if __name__ == '__main__':

    """
    Creating kinase large-scale datasets from Papyrus
    
    WARNING : This script takes a long time to run (around 5h) 
    """

    mkdirs('data/datasets')

    # Load Papyrus data
    print('Downloading Papyrus data...')
    download_papyrus(version='05.6', only_pp=False, descriptors=False, structures=False, outdir='data')
    
    # Retrieve, filter & pivot data
    all_kinase_data = retrieve_kinase_data_from_Papyrus(source_path='data', version='05.6', plusplus=False)
    kinase200, kinase1000 = filter_data(all_kinase_data)
    kinase200.to_csv('data/kinase200.csv.gz', index=False)
    kinase1000.to_csv('data/kinase1000.csv.gz', index=False)

    # Kinase 200 ###############################################################                                                                                                   
    print('Processing Kinase 200 dataset...')
    kinase200 = pd.read_csv('data/kinase200.csv.gz')
    kinase200 = pivot_Papyrus_data(kinase200)
    targets_kinase200 = kinase200.columns[2:].tolist()
    print(f'Number of targets: {len(targets_kinase200)}')

    rges_kinase200 = random_global_equilibrated_random_split(kinase200, targets_kinase200, seed=SEED)
    rges_kinase200 = compute_intersubset_Tanimoto_distance(rges_kinase200, n_jobs=-1)
    rges_kinase200.to_csv('data/datasets/kinase200_RGES.csv.gz', index=False)
    create_input_files('data/datasets/kinase200_RGES.csv.gz', 'ModelInputs/kinase200/RGES/Original')

    dgbc_kinase200 = dissimilaritydrive_global_balanced_cluster_split(kinase200, targets_kinase200, sizes=[0.8,0.1,0.1] )
    dgbc_kinase200 = compute_intersubset_Tanimoto_distance(dgbc_kinase200, n_jobs=16)
    dgbc_kinase200.to_csv('data/datasets/kinase200_DGBC.csv.gz', index=False)
    create_input_files('data/datasets/kinase200_DGBC.csv.gz', 'ModelInputs/kinase200/DGBC/Original')


    # Kinase 1000 ##############################################################
    print('Processing Kinase 1000 dataset...')
    kinase1000 = pd.read_csv('data/kinase1000.csv.gz')
    kinase1000 = pivot_Papyrus_data(kinase1000)
    targets_kinase1000 = kinase1000.columns[2:].tolist()

    rges_kinase1000 = random_global_equilibrated_random_split(kinase1000, targets_kinase1000, seed=SEED)
    rges_kinase1000 = compute_intersubset_Tanimoto_distance(rges_kinase1000)
    rges_kinase1000.to_csv('data/datasets/kinase1000_RGES.csv.gz', index=False)
    create_input_files('data/datasets/kinase1000_RGES.csv.gz', 'ModelInputs/kinase1000/RGES/Original')
    
    dgbc_kinase1000 = dissimilaritydrive_global_balanced_cluster_split(kinase1000, targets_kinase1000, sizes=[0.8,0.1,0.1] )
    dgbc_kinase1000 = compute_intersubset_Tanimoto_distance(dgbc_kinase1000, n_jobs=16)
    dgbc_kinase1000.to_csv('data/datasets/kinase1000_DGBC.csv.gz', index=False)
    create_input_files('data/datasets/kinase1000_DGBC.csv.gz', 'ModelInputs/kinase1000/DGBC/Original')







