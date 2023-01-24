import os
import sys
import tqdm
import json

import pandas as pd

from chemprop.args import TrainArgs, PredictArgs
from chemprop.train import cross_validate, run_training, make_predictions

from .utils import mkdirs

def train_chemprop_STs(data_path : str, valid_path : str, test_path : str, model_path : str,  param_path : str = None):

    """
    Train a chemprop model for each kinase in the dataset.
    
    Parameters
    ----------
    data_path : str
        Path to the training data.
    valid_path : str
        Path to the validation data.
    test_path : str
        Path to the test data.
    model_path : str
        Path to the directory where the models will be saved.
    param_path : str, optional
        Path to the json file containing the parameters for the model, by default None
    """
    
    mkdirs(model_path)
    kinases = pd.read_csv(data_path).drop(['SMILES', 'InChIKey', 'Split', 'Subset', 'MinInterSetTd'], axis=1, errors='ignore').columns.tolist()

    cmd = '--data_path {} '.format(data_path)
    cmd += '--separate_val_path {} '.format(valid_path)
    cmd += '--separate_test_path {} '.format(test_path)

    cmd += '--dataset_type regression '
    cmd += '--smiles_columns SMILES '
    cmd += '--metric rmse '
    cmd += '--extra_metrics r2 '
    cmd += '--quiet '
    cmd += '--gpu 0 '

    if param_path :
        with open(param_path) as d: params = json.load(d)
        for k, v in params.items(): cmd += f'--{k} {v} '

    for kinase in tqdm.tqdm(kinases):
        if os.path.exists(f'{model_path}/{kinase}'): continue
        cmd += '--target_columns {} '.format(kinase)
        cmd += '--save_dir {}/{} '.format(model_path, kinase)
        args = TrainArgs().parse_args(cmd.split()) 
        cross_validate(args=args, train_func=run_training)

def train_chemprop_MT(data_path : str, valid_path : str, test_path : str, model_path : str, param_path : str = None, **kwargs):
    
    """
    Train a chemprop model for the whole dataset.
    
    Parameters
    ----------
    data_path : str
        Path to the training data.
    valid_path : str
        Path to the validation data.
    test_path : str
        Path to the test data.
    model_path : str
        Path to the directory where the models will be saved.
    param_path : str, optional
        Path to the json file containing the parameters for the model, by default None
    """

    mkdirs(model_path)

    cmd = '--data_path {} '.format(data_path)
    cmd += '--separate_val_path {} '.format(valid_path)
    cmd += '--separate_test_path {} '.format(test_path)

    cmd += '--dataset_type regression '
    cmd += '--smiles_columns SMILES '
    cmd += '--metric rmse '
    cmd += '--extra_metrics r2 '
    cmd += '--quiet '
    cmd += '--gpu 0 '

    if param_path :
        with open(param_path) as d: params = json.load(d)
        print(params)
        for k, v in params.items(): 
            cmd += f'--{k} {v} '

    if kwargs:
        for k, v in kwargs.items(): cmd += f'--{k} {v} '

    cmd += '--save_dir {} '.format(model_path)

    print(cmd)
    args = TrainArgs().parse_args(cmd.split()) 
    cross_validate(args=args, train_func=run_training)

def predict_chemprop_MT(data_path : str, model_path : str, preds_path : str):

    """
    Predict the activity of a dataset using a chemprop multi-task model.

    Parameters
    ----------
    data_path : str
        Path to the data.
    model_path : str
        Path to the directory where the models are saved.
    preds_path : str
        Path to the file where the predictions will be saved.
    
    Returns
    -------
    preds : pd.DataFrame
        Dataframe containing the predictions.
    """

    mkdirs(os.path.dirname(preds_path))
    
    cmd = '--test_path {} '.format(data_path)
    cmd += '--checkpoint_path {}/fold_0/model_0/model.pt '.format(model_path)
    cmd += '--preds_path {} '.format(preds_path)

    make_predictions(args=PredictArgs().parse_args(cmd.split()))

def predict_chemprop_STs(data_path : str, model_path : str, preds_path : str):

    """
    Predict the activity per target of a dataset using a chemprop single-task model.

    Parameters
    ----------
    data_path : str
        Path to the data.
    model_path : str
        Path to the directory where the models are saved.
    preds_path : str
        Path to the file where the predictions will be saved.
    
    Returns
    -------
    preds : pd.DataFrame
        Dataframe containing the predictions.
    """
    

    targets = pd.read_csv(data_path).drop(['SMILES', 'InChIKey', 'Split', 'Subset', 'MinInterSetTd'], axis=1, errors='ignore').columns.tolist()
    preds = pd.read_csv(data_path)
    
    for target in tqdm.tqdm(targets):
        cmd = '--test_path {} '.format(data_path)
        cmd += '--checkpoint_path {}/{}/fold_0/model_0/model.pt '.format(model_path, target)
        cmd += '--preds_path {} '.format('tmp.csv')
        cmd += '--empty_cache '
        make_predictions(args=PredictArgs().parse_args(cmd.split()))
        preds[target] = pd.read_csv('tmp.csv')[target]
    
    mkdirs(os.path.dirname(preds_path))
    preds.to_csv(preds_path, index=False)
    

