import re
import json
import requests
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from typing import List

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem

from Bio import Entrez
from Bio.Entrez import efetch
Entrez.email = 'A.N.Other@example.com'

from papyrus_scripts.reader import read_papyrus, read_protein_set
from papyrus_scripts.preprocess import keep_protein_class, keep_quality, keep_type, consume_chunks

from chembl_webresource_client.new_client import new_client

def retrieve_kinase_data_from_Papyrus(source_path : str, version : str = '05.6', plusplus : bool = True):

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

    if plusplus:
        mol_data = read_papyrus(is3d=False, chunksize=100000, source_path=source_path, plusplus=plusplus, version=version)
        protein_data = read_protein_set(source_path=source_path)
        filter_protein = keep_protein_class(data=mol_data, protein_data=protein_data, classes=[{'l3': 'Protein Kinase'}])
        all_kinase_data = consume_chunks(filter_protein, progress=True, total=5)
    else:
        mol_data = read_papyrus(is3d=False, chunksize=1000000, source_path=source_path, plusplus=plusplus, version=version)
        protein_data = read_protein_set(source_path=source_path)
        filter_quality = keep_quality(data=mol_data, min_quality='high')
        filter_protein = keep_protein_class(data=filter_quality, protein_data=protein_data, classes=[{'l3': 'Protein Kinase'}])
        filter_type = keep_type(data=filter_protein, activity_types=['Ki', 'IC50', 'KD', 'EC50'])
        all_kinase_data = consume_chunks(filter_type, progress=True, total=60)

    print('Number of kinase targets: {}'.format(all_kinase_data.target_id.nunique()))
    print('Number of activity points before filtering: {}'.format(all_kinase_data.shape[0]))
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
    data = data[data.pchembl_value_StdDev < 1 ].reset_index(drop=True)
    print(f'[1/6] Number of activity points after removing datapoints with std > 1: {data.shape[0]}')

    # Remove compounds with MW > 1000 Da
    data = data[data.SMILES.apply(Chem.MolFromSmiles).apply(Descriptors.MolWt) < 1000 ].reset_index(drop=True)
    print(f'[2/6] Number of activity points after removing molecules with MW > 1000: {data.shape[0]}')

    # Remove allosteric compounds
    data = filter_allosteric_compounds(data)

    # Remove kinase with less than 200 data points
    targets = data.target_id.unique()
    targets = [ target for target in targets if data.target_id.value_counts()[target] >= 200 ]
    data200 = data[data.target_id.isin(targets)].reset_index(drop=True)
    print(f'Number of targtes and activity points after removing targets with < 200 data points: {len(targets)}, {data200.shape[0]}')

    # Remove kinase with less than 1000 data points
    targets = [ target for target in targets if data.target_id.value_counts()[target] >= 1000 ]
    data1000 = data[data.target_id.isin(targets)].reset_index(drop=True)
    print(f'Number of targtes and activity points after removing targets with < 1000 data points: {len(targets)}, {data1000.shape[0]}')

    return data200, data1000

def abstract_parser(document_ids : List, keyword_list : List):

    """ 
    Parse abstracts from PubMed, PubChem and Crossref
    for binding type keywords
    
    Parameters:
        document_ids (list of str) : list of document IDs
        keyword_list (list of str) : list of keywords to search for in abstracts

    Returns:
        selected_abstracts (list of str) : list of document IDs (PMID) with abstracts containing keywords
    """

    # Get, parse and annotate abstracts
    selected_abstracts = []
    for doc_id in tqdm(document_ids, desc='Parsing abstracts'):
        try :
            if 'PMID' in doc_id:
                pmid = doc_id[5:]
                handle = efetch(db='pubmed', id=pmid, retmode='text', rettype='abstract')
                abstract = handle.read().lower()
            elif 'PubChemAID' in doc_id:
                aid = doc_id[11:]
                handle = efetch(db='pcassay', id=aid, retmode='text', rettype='abstract')
                abstract = handle.read().lower()
            elif 'DOI' in doc_id:
                try:
                    url = f'https://api.crossref.org/works/{doc_id[4:]}'
                    r = requests.get(url)
                    crossref = r.json()
                except json.decoder.JSONDecodeError:
                    continue
                abstract = crossref['message']['abstract'].lower()
            else:
                continue
        except:
            print("Couldn't parse the abstract from {}".format(doc_id))
            continue

        for keyword in keyword_list:
            if keyword in abstract:
                selected_abstracts.append(doc_id)
                break
    
    return selected_abstracts

def chembl_description_parser(assay_ids: List, keyword_list: List):

    """ Parse chembl assay descriptions for binding type keywords
    
    Parameters:
        assay_ids (list of str) : list of assay IDs
        keyword_list (list of str) : list of keywords to search for in assay descriptions
        
    Returns:
        selected_assays (list of str) : list of assay IDs with descriptions containing keywords
    """

    assay = new_client.assay
    descriptions = assay.filter(assay_id__in=assay_ids).only(['description'])
    selected_assays = []
    for assay_id, description in tqdm(zip(assay_ids, descriptions), total=len(assay_ids), desc='Parsing assay descriptions'):
        description = description['description']
        if description is not None:
            description = description.lower()
            for keyword in keyword_list:
                if keyword in description:
                    selected_assays.append(assay_id)
                    break
    
    return selected_assays

def patent_parser(pantents : List, keywords : List):

    selected_abstracts = []
    for patent in tqdm(pantents, desc='Parsing patents'):
        url = f'https://patents.google.com/patent/{patent[7:]}/en'
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')
        meta = soup.find_all("meta")

        for m in meta:
            if m.get("name") == "description":
                abstract = m.get("content").lower()
        # remove newlines and replace all blancs of any lenght with a single space
        abstract = re.sub(r' +', ' ', abstract.replace('\n', ' '))

        for keyword in keywords:
            if keyword in abstract:
                selected_abstracts.append(patent)
                break

    return selected_abstracts


def filter_allosteric_compounds(data):

    """
    Filter out allosteric compounds in 3 steps:

    1. Filter out compounds with a binding type keyword in the chembl assays description
    2. Filter out compounds with a binding type keyword in the abstracts (from PubMed or PubChem)
    3. Filter out compounds with a binding type keyword in the patents
    4. Filter out compounds with maximum Tanimoto similarity > 0.9 to compounds assigned to be allosteric in step 1,2 or 3
    
    Parameters
    ----------
    data : pd.DataFrame
        Papyrus data
    
    Returns
    -------
    data : pd.DataFrame
        Filtered Papyrus data
    """

    allosteric_keywords = [
        'activators',
        'allosteric',
        'allosterism',
        'allostery',
        'alosteric',
        'alostery',
        'alosterism',
        'indirect activation',
        'indirectly activate'
        'indirectly inhibit',
        'indirectly modulate',
        'negative modulator',
        'negative modulators',
        'nnrti', #non-nucleoside reverse transcriptase inhibitor
        'non-competitive',
        'non-nucleoside reverse transcriptase inhibitor',
        'non-substrate',
        'noncompetitive',
        'nonsubstrate',
        'positive modulator',
        'positive modulators',
        'receptor modulator',
        'regulatory site',
        'secondary binding site',
        'secondary pocket',
        'un-competitive',
        'uncompetitive',
        'pif', #PIF-binding pocket
        'myristoyl', #myristoyl pocket
        'pseudo-kinase',
        'pseudokinase',


    ]

    data_allo = data.copy()

    # 1. Filter out compounds with a binding type keyword in the chembl assays description    
    # Get all chembl assay IDs
    chembl_assay_ids = []
    for aids in data.AID.unique():
        for aid in aids.split(';'):
            if 'CHEMBL' in aid:
                chembl_assay_ids.append(aid)
    chembl_assay_ids = list(set(chembl_assay_ids))

    # Parse chembl assay descriptions
    selected_assays = chembl_description_parser(chembl_assay_ids, allosteric_keywords)

    # Drop allosteric compounds
    allosteric_smiles = []
    for smiles, aids in zip(data.SMILES, data.AID):
        for aid in aids.split(';'):
            if aid in selected_assays:
                allosteric_smiles.append(smiles)
                break
    data = data[~data.SMILES.isin(allosteric_smiles)].reset_index(drop=True)
    print(f'[3/6] Number of activity points after removing allosteric compounds (ChEMBL assay descriptions): {data.shape[0]}')

    # 2. Filter out compounds with a binding type keyword in the abstracts (from PubMed or PubChem)
    # Get all PubChemAIDs, PMIDs and DOIs from document IDs
    parsable_docs = []
    for doc_ids in data.all_doc_ids.unique():
        for doc_id in doc_ids.split(';'):
            if 'PMID' in doc_id or 'PubChemAID' in doc_id or 'DOI' in doc_id:
                parsable_docs.append(doc_id)
    parsable_docs = list(set(parsable_docs))

    # Parse abstracts
    selected_doc_ids = abstract_parser(parsable_docs, allosteric_keywords)

    # Drop allosteric compounds
    for smiles, doc_ids in zip(data.SMILES, data.all_doc_ids):
        for doc_id in doc_ids.split(';'):
            if doc_id in selected_doc_ids:
                allosteric_smiles.append(smiles)
                break
    data = data[~data.SMILES.isin(allosteric_smiles)].reset_index(drop=True)
    print(f'[4/6] Number of activity points after removing allosteric compounds (abstracts): {data.shape[0]}')

    # 3. Filter out compounds with a binding type keyword in the patents
    # Get all patent IDs
    patent_ids = []
    for doc_ids in data.all_doc_ids.unique():
        for doc_id in doc_ids.split(';'):
            if 'PATENT' in doc_id:
                patent_ids.append(doc_id)
    patent_ids = list(set(patent_ids))

    # Parse patents
    selected_patents = patent_parser(patent_ids, allosteric_keywords)

    # Drop allosteric compounds
    for smiles, doc_ids in zip(data.SMILES, data.all_doc_ids):
        for doc_id in doc_ids.split(';'):
            if doc_id in selected_patents:
                allosteric_smiles.append(smiles)
                break
    data = data[~data.SMILES.isin(allosteric_smiles)].reset_index(drop=True)
    print(f'[5/6] Number of activity points after removing allosteric compounds (patents): {data.shape[0]}')


    # 4. Drop compounds similar to allosteric compounds
    if len(allosteric_smiles) == 0:
        return data
    
    allosteric_smiles = list(set(allosteric_smiles))
    other_smiles = data['SMILES'].unique()
    allosteric_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, nBits=2048) for smiles in allosteric_smiles]
    other_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, nBits=2048) for smiles in other_smiles]

    for smiles, fp in zip(other_smiles, other_fps):
        max_tanimoto = np.max(DataStructs.BulkTanimotoSimilarity(fp, allosteric_fps))
        if max_tanimoto > 0.8:
            data = data[data.SMILES != smiles]
            allosteric_smiles.append(smiles)

    data = data.reset_index(drop=True)
    print(f'[6/6] Number of activity points after removing compounds similar to allosteric compounds: {data.shape[0]}')

    data_allo = data_allo[data_allo.SMILES.isin(allosteric_smiles)].reset_index(drop=True)
    data_allo.to_csv('data/kinase_allosteric.csv.gz', index=False)
    print(f'Allotseric compounds saved to data/kinase_allosteric.csv.gz')
    
    return data

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

    targets = data.target_id.unique()

    data = data.pivot(index=['SMILES', 'InChIKey'], columns='target_id', values = 'pchembl_value_Mean').reset_index()
    print(f'Number of compounds: {data.shape[0]}')
    print(f'Number of targets: {len(targets)}')

    # Get number of datapoints (i.e. non NaN cells) and density
    

    ndatapoints = 0
    for target in targets:
        ndatapoints += data[target].dropna().count()
    print(f'Number of datapoints: {ndatapoints}')
    print(f'Desity: {ndatapoints / (data.shape[0] * len(targets))*100:.2f}%')

    return data