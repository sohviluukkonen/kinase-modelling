import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from papyrus_scripts.download import download_papyrus
from papyrus_scripts.reader import read_papyrus, read_protein_set
from papyrus_scripts.preprocess import keep_quality, keep_type, keep_protein_class
from papyrus_scripts.preprocess import consume_chunks

def load_papyrus_data(papyrus_path : str):

    """
    Download all activity datapoints from Papyrus v05.5
    
    Parameters
    ----------
    papyrus_path : str
        Path to Papyrus data
    """

    print('Downloading Papyrus data...')

    download_papyrus(version='05.5', only_pp=False, descriptors=False, outdir=papyrus_path)

def retrive_kinase_data_from_Papyrus(source_path : str):

    """
    Filter kinase Ki and IC50 data from Papyrus v05.5
    
    Parameters
    ----------
    source_path : str
        Path to Papyrus data

    Returns
    -------
    all_kinase_data : pd.DataFrame
        Filtered kinase data
    """

    print('Retrive kinase data from Papyrus...')

    mol_data = read_papyrus(is3d=False, chunksize=100000, source_path=source_path)
    protein_data = read_protein_set(source_path=source_path)
    filter_quality = keep_quality(data=mol_data, min_quality='medium')
    filter_protein = keep_protein_class(data=filter_quality, protein_data=protein_data, classes=[{'l2': 'Kinase'}])
    filter_type = keep_type(data=filter_protein, activity_types=['IC50','Ki'])
    all_kinase_data = consume_chunks(filter_type, progress=False, total=60)

    print(f'Number of activity points: {all_kinase_data.shape[0]}')
    return all_kinase_data

def filter_data(data: pd.DataFrame):

    """
    Filter Papyrus data:
    - Remove bioactivity values with multiple data points if std >= 1
    - Remove compounds with MW > 1000 Da
    - Remove kinase with less than 200 data points --> data200
    - Remove kinase with less than 1000 data points --> data1000

    Parameters
    ----------
    data : pd.DataFrame
        Papyrus data
    
    Returns
    -------
    data200 : pd.DataFrame
        Filtered data with at least 200 datapoints per kinase
    data1000 : pd.DataFrame
        Filtered data with at least 1000 datapoints per kinase
    """


    print('Filter data...')

    # Convert all smiles to canonical ones
    data['SMILES'] = data['SMILES'].apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles)

    # Remove bioactivity values with multiple data points if std >= 1
    data = data[data.pchembl_value_StdDev < 1 ]
    print(f'Number of activity points after removing datapoints with std > 1: {data.shape[0]}')

    # Remove compounds with MW > 1000 Da
    data = data[data.SMILES.apply(Chem.MolFromSmiles).apply(Descriptors.MolWt) < 1000 ]
    print(f'Number of activity points after removing molecules with MW > 1000: {data.shape[0]}')

    # Remove kinase with less than 200 data points
    targets = data.target_id.unique()
    targets = [ target for target in targets if data.target_id.value_counts()[target] > 200 ]
    data200 = data[data.target_id.isin(targets)]
    print(f'Number of targtes and activity points after removing targets with < 200 datapoits: {len(targets)}, {data.shape[0]}')

    # Remove kinase with less than 1000 data points
    targets = [ target for target in targets if data.target_id.value_counts()[target] > 1000 ]
    data1000 = data[data.target_id.isin(targets)]
    print(f'Number of targtes and activity points after removing targets with < 1000 datapoits: {len(targets)}, {data.shape[0]}')

    return data200, data1000

def pivot_Papyrus_data(data):

    """Pivot Papyrus data to have one row per compound and one column per target
    
    Parameters
    ----------
    data : pd.DataFrame
        Papyrus data
    
    Returns
    -------
    pd.DataFrame
        Pivoted data
    """

    print('Pivot data...')

    return data.pivot(index=['SMILES', 'InChIKey'], columns='target_id', values = 'pchembl_value_Mean').reset_index()