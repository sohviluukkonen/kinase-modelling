import os
import sys

import numpy as np
import pandas as pd

from src.chemprop import train_chemprop_MT, train_chemprop_STs, predict_chemprop_MT, predict_chemprop_STs
from src.rf import train_RF_STs, predict_RF_STs, train_optim_RF_STs
from src.xgb import train_XGB_STs, predict_XGB_STs, train_optim_XGB_STs
from src.utils import mkdirs
from src.inputs import create_imputed_inputs

# Set random seed
seed = 2022
import torch
torch.manual_seed(seed)
import random
random.seed(seed)
np.random.seed(seed)


def run_model(model :str, 
    dataset : str = 'kinase200', 
    split : str = 'Random', 
    mode : str = 'ST', 
    params : str = 'Default',
    param_path : str = None,
    imputation : str = None):

    """
    Run a model on a dataset.
    
    Parameters
    ----------
    model : str
        Name of the model to run.
    dataset : str
        Name of the dataset to use.
    split : str
        Name of the split to use.
    mode : str
        Name of the mode to use: ST or MT.
    params : str
        Name of the parameters to use.
    param_path : str
        Path to the parameters to use.
    imputation : str
        Name of the imputation method to use.
    """

    # If param_path is given, set params to HyperOpt istaed of the name of the method
    params = 'HyperOpt' if param_path else params
    
    # If an imputation method is given, change model name based on imputation method
    if imputation : 
        if mode == 'ST' : sys.exit('Imputation can only be used with multitask models')
        else: model = model + '_Imputed' + imputation
    
    # Create paths
    data_path = f'ModelInputs/{dataset}/{split}'
    model_path = f'Models/{model}/{mode}/{dataset}/{split}/{params}'
    preds_path = f'Predictions/{model}/{mode}/{dataset}/{split}/{params}/predictions.csv'

    # Modify data_path depending on imputation method
    if imputation : data_path = f'{data_path}/Imputed{imputation}'
    else : data_path = f'{data_path}/Original'

    print(f'Data path: {data_path}')
    print(f'Model path: {model_path}')
    print(f'Predictions path: {preds_path}')
    
    # Random Forest
    if model == 'RF':
        if mode == 'MT' or imputation != None:
            sys.exit('For RF, only ST model is implemented')
        elif params == 'Default':
            train_RF_STs(f'{data_path}/train.csv', model_path, )
            predict_RF_STs(f'{data_path}/test.csv', model_path, preds_path)
        else:
            train_optim_RF_STs(f'{data_path}/train.csv', f'{data_path}/valid.csv', model_path)
            predict_RF_STs(f'{data_path}/test.csv', model_path, preds_path)

    # XGBoost
    elif model == 'XGB':
        if mode == 'MT' or imputation != None:
            sys.exit('For XGB, only ST model is implemented')
        elif params == 'Default':
            train_XGB_STs(f'{data_path}/train.csv', model_path)
            predict_XGB_STs(f'{data_path}/test.csv', model_path, preds_path)
        else:
            train_optim_XGB_STs(f'{data_path}/train.csv', f'{data_path}/valid.csv', model_path)
            predict_XGB_STs(f'{data_path}/test.csv', model_path, preds_path)

    # Chemprop
    elif model.startswith('CP'):
        if mode == 'ST':
            train_chemprop_STs(f'{data_path}/train.csv', f'{data_path}/valid.csv', f'{data_path}/test.csv', model_path, param_path)
            predict_chemprop_STs(f'{data_path}/test.csv', model_path, preds_path)   
        else:
            train_chemprop_MT(f'{data_path}/train.csv', f'{data_path}/valid.csv', f'{data_path}/test.csv', model_path, param_path)
            predict_chemprop_MT(f'{data_path}/test.csv', model_path, preds_path)   

if __name__ == '__main__':

    for dataset in ['kinase1000', 'kinase200']:
        for split in  ['RGES', 'DGBC']:

            # With default parameters
            
            # Single-task Random Forest
            run_model('RF', dataset, split, 'ST')
            # Single-task XGBoost
            run_model('XGB', dataset, split, 'ST')
            # Single-task Chemprop
            run_model('CP', dataset, split, 'ST')
            # Multi-task Chemprop
            run_model('CP', dataset, split, 'MT')

            # Multi-task Chemprop - Mean imputed
            create_imputed_inputs(f'Preprocessing/{dataset}_{split}.csv', f'Datasets/{dataset}/{split}/ImputedMean', 'Mean')
            run_model('CP', dataset, split, 'MT', imputation='Mean')

            # Multi-task Chemprop - RF imputed
            create_imputed_inputs(f'Preprocessing/{dataset}_{split}.csv', f'Datasets/{dataset}/{split}/ImputedRF',
                'RF', f'Models/RF/ST/{dataset}/{split}/Default/'
                )
            run_model('CP', dataset, split, 'MT', imputation='RF')
            
            # With optimized parameters
            if dataset == 'kinase200':
                run_model('RF', dataset, split, 'ST', 'HyperOpt')
                run_model('XGB', dataset, split, 'ST', 'HyperOpt')
                run_model('CP', dataset, split, 'ST', param_path='chemprop_hyperparams.json')
                run_model('CP', dataset, split, 'MT', param_path='chemprop_hyperparams.json')
                run_model('CP', dataset, split, 'MT', param_path='chemprop_hyperparams.json', imputation='Mean')        
                run_model('CP', dataset, split, 'MT', param_path='chemprop_hyperparams.json', imputation='RF')