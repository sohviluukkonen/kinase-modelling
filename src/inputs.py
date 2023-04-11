import numpy as np
import pandas as pd

from .utils import mkdirs
from .rf import predict_RF_STs 

def create_input_files(in_file : str, out_path : str):

    mkdirs(out_path)

    data = pd.read_csv(in_file)
    for subset in ['train', 'valid', 'test']:
        df = data[data.Subset == subset].drop(['InChIKey', 'Split', 'Subset', 'MinInterSetTd'], axis=1, errors='ignore')
        df.to_csv(f'{out_path}/{subset}.csv', index=False)
    df = data.drop(['InChIKey', 'Split', 'Subset', 'MinInterSetTd'], axis=1, errors='ignore')
    df.to_csv(f'{out_path}/all.csv', index=False)   

def create_imputed_inputs(in_file : str, out_path : str, mode : str = 'RF', model_path = None):

    mkdirs(out_path)
    data = pd.read_csv(in_file)
    targets = data.drop(['SMILES', 'InChIKey', 'Split', 'Subset', 'MinInterSetTd'], axis=1, errors='ignore').columns.tolist()

    if mode == 'Mean':
        # mean = data.loc[:, data.columns.isin(targets)].mean(axis=1)
        # for i in range(len(mean)):
        #     data.iloc[i] = data.iloc[i].fillna(mean[i])

        # Mean value per row and column
        mean_row = data.loc[:, data.columns.isin(targets)].mean(axis=1)
        mean_col = data.loc[:, data.columns.isin(targets)].mean(axis=0)

        # Combination of row and column means
        mean = np.zeros((len(mean_row), len(mean_col)))
        for i in range(len(mean_row)):
            for j in range(len(mean_col)):
                mean[i,j] = (mean_row[i] + mean_col[j])/2
        mean = pd.DataFrame(mean, columns=targets)
        mean['SMILES'] = data['SMILES']

        # Fill missing values with mean
        data.fillna(mean, inplace=True)

    if mode == 'RF': 
        preds = predict_RF_STs(in_file, model_path)
        for target in targets:
            inan = data[target].isna() 
            data.loc[inan, target] = preds.loc[inan, target]

    for subset in ['train', 'valid', 'test']:
        df = data[data.Subset == subset].drop(['InChIKey', 'Split', 'Subset', 'MinInterSetTd'], axis=1, errors='ignore')
        df.to_csv(f'{out_path}/{subset}.csv', index=False)
    data.to_csv(f'{out_path}/all.csv', index=False)

def create_input_files_from_pQSARdatasets():

    """ 
    Create inputs for the model from pQSAR2.0 paper's raw datasets
    """

    for split in ['Random', 'Realistic']:
        
        input_prefix = f'data/pQSARdata/{split}Split'
        output_dir = f'ModelInputs/Assays/{split}'
        mkdirs(output_dir)

        # Train data
        subset = 'train'
        df = pd.read_csv(f'{input_prefix}_{subset}.csv.gz').rename(columns={'smiles': 'SMILES'})
        # Experimental data
        train_exp = df.drop(['ID', 'pIC50_pred'], axis=1).pivot(index='SMILES', columns='AssayID', values='pIC50_exp').reset_index()
        train_exp.to_csv(f'{output_dir}/exp_{subset}.csv', index=False)
        # pQSAR2.0 Predicted data
        train_pred = df.drop(['ID', 'pIC50_exp'], axis=1).pivot(index='SMILES', columns='AssayID', values='pIC50_pred').reset_index()
        train_pred.to_csv(f'{output_dir}/pred_{subset}.csv', index=False)
        print(len(df), len(train_exp), len(train_pred))

        # Test data
        subset = 'test'
        df = pd.read_csv(f'{input_prefix}_{subset}.csv').rename(columns={'smiles': 'SMILES'})
        # Experimental data
        test_exp = df.drop(['ID', 'pIC50_pred'], axis=1).pivot(index='SMILES', columns='AssayID', values='pIC50_exp').reset_index()
        test_exp.to_csv(f'{output_dir}/exp_{subset}.csv', index=False)
        # pQSAR2.0 Predicted data
        test_pred = df.drop(['ID', 'pIC50_exp'], axis=1).pivot(index='SMILES', columns='AssayID', values='pIC50_pred').reset_index()
        test_pred.to_csv(f'{output_dir}/pred_{subset}.csv', index=False)
        print(len(df), len(test_exp), len(test_pred))

        # Combine train and test data
        exp = pd.concat([train_exp, test_exp], axis=0)
        pred = pd.concat([train_pred, test_pred], axis=0)
        # Merge unique SMILES
        exp = exp.groupby('SMILES').max().reset_index()
        pred = pred.groupby('SMILES').max().reset_index()
        # Save merged data
        exp.to_csv(f'{output_dir}/exp_all.csv', index=False)
        pred.to_csv(f'{output_dir}/pred_all.csv', index=False)
