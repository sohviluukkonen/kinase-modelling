import os
import tqdm
import joblib

import cupy as cp
import numpy as np
import pandas as pd

from .utils import mkdirs, compute_fps

# strategies to deal with multiple outputs
from py_boost.multioutput.sketching import *
from py_boost.multioutput.target_splitter import *

from py_boost import GradientBoosting # basic GradientBoosting class
from py_boost.gpu.losses import * # utils for the custom loss
from py_boost.multioutput.sketching import * # utils for multioutput
from py_boost.multioutput.sketching import RandomProjectionSketch

from sklearn.model_selection import ParameterSampler
from sklearn.metrics import make_scorer, r2_score

from scipy.stats import rv_continuous, rv_discrete

# Class to handle missing values in a PyBoost multi-task model ################

class MSEWithNanLoss(MSELoss):
    """
    This is custom MSE Loss that accepts NaN values and ignores features
    """
    def __init__(self, ):
        
        self.feats_cols = None
    
    def get_grad_hess(self, y_true, y_pred):
        """
        
        Args:
            y_true: cp.ndarray of target values
            y_pred: cp.ndarray of predicted values
            
        Returns:

        """
        mask = ~cp.isnan(y_true)
        # apply features mask
        grad = y_pred - cp.where(mask, y_true, 0)
        hess = mask.astype(cp.float32)
        grad *= hess
        # we will ignore not only NaNs but also columns that are used as features !!!
        if self.feats_cols is not None:
            hess[:, self.feats_cols] = 0
            grad *= hess
        
        return grad, hess

    def base_score(self, y_true):
        """This method defines how to initialize the ensemble
        
        Args:
            y_true: cp.ndarray of target values
            
        Returns:

        """
        return cp.nanmean(y_true, axis=0)

class RMSEWithNaNMetric(RMSEMetric):
    """
    This is custom MSE Loss that accepts NaN values and ignores features
    """
    def __init__(self, target_cols):
        """
        
        Args:
            target_cols: list of int, indices of columns that could be both features and targets
            
        Returns:

        """
        self.target_cols = target_cols

    
    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        
        Args:
            y_true: cp.ndarray of target values
            y_pred: cp.ndarray of predicted values
            sample_weight: cp.nndarray of sample weights or None
            
        Returns:

        """
        y_true = y_true[:, self.target_cols]
        y_pred = y_pred[:, self.target_cols]         
        
        mask = ~cp.isnan(y_true)
        
        err = (cp.where(mask, y_true, 0) - y_pred) ** 2
        return err[mask].mean() ** .5

# Function to create and train a model

def PB_hyperparams():
    """
    Returns a dictionary of hyperparameters for the PyBoost model
    """

    return {
        "lr":  [0.005, 0.01, 0.05],
        "max_depth": [8, 10, 12],
        "lambda_l2": [5, 10, 20, 50],
        "colsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_gain_to_split": [0, 1, 2],
        "gd_steps": [1, 2, 5, 10],
        "quantization": ['Uniform', 'Uniquant']
    }



def train_PB_MT(
    data_path : str,
    model_path : str,
    seed : int = 2022,
    ):

    mkdirs(model_path)

    data = pd.read_csv(data_path)
    targets = data.drop(['SMILES', 'InChIKey', 'Split', 'Subset', 'MinInterSetTd', 'set_index'], axis=1, errors='ignore').columns.tolist()    
    target_index = np.arange(len(targets))
    
    X_train = compute_fps(data).astype(float).values
    y_train = data[targets] #.values

    loss = MSEWithNanLoss()
    metric = RMSEWithNaNMetric(target_index)
    sketch = RandomProjectionSketch(1)

    model = GradientBoosting(
        loss, metric=metric, seed=seed, multioutput_sketch=sketch,
        )

    model.fit(X_train, y_train) #, eval_sets=eval_sets)

    joblib.dump(model, model_path + '/model.joblib')

def train_optim_PB_MT(
    data_path : str,
    val_path : str,
    model_path : str,
    seed : int = 2022,
    n_iter : int = 100,
    ):

    mkdirs(model_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    data = pd.read_csv(data_path)
    data_val = pd.read_csv(val_path)
    targets = data.drop(['SMILES', 'InChIKey', 'Split', 'Subset', 'MinInterSetTd', 'set_index'], axis=1, errors='ignore').columns.tolist()
    target_index = np.arange(len(targets))
    
    X_train = compute_fps(data).astype(float).values
    y_train = data[targets]


    X_val = compute_fps(data_val).astype(float).values
    y_val = data_val[targets]
    eval_sets = [{'X': X_val, 'y': y_val}]

    if 'kinase1000' in model_path:
        # With parameters optimized for the Kinase200 datasets
        if 'RGES' in model_path:
            param_list = [
                {'subsample': 0.8, 'quantization': 'Uniform', 'min_gain_to_split': 1, 'max_depth': 12, 'lr': 0.005, 'lambda_l2': 50, 'gd_steps': 10, 'colsample': 0.8}
            ]
        else:
            param_list = [
                {'subsample': 0.7, 'quantization': 'Uniquant', 'min_gain_to_split': 2, 'max_depth': 10, 'lr': 0.05, 'lambda_l2': 20, 'gd_steps': 1, 'colsample': 0.7}
            ]
    else:
        param_list = list(ParameterSampler(PB_hyperparams(), n_iter=n_iter, random_state=seed))

    loss = MSEWithNanLoss()
    metric = RMSEWithNaNMetric(target_index)
    sketch = RandomProjectionSketch(1)

    best_score = -np.inf
    best_params = None
    best_model = None
  
    for j, params in tqdm.tqdm(enumerate(param_list), total=len(param_list)):

        print('Training model', j, 'with params', params)

        model = GradientBoosting(
            loss, metric=metric, seed=seed, multioutput_sketch=sketch, verbose=1000,
            ntrees=10000, es=250, **params
            )
        
        try :
            model.fit(X_train, y_train, eval_sets=eval_sets)
            y_pred = model.predict(X_val)
            y_pred = pd.DataFrame(y_pred, columns=targets)
        except:
            print('Warning: Failed to fit the model or predict on validation set')
            # Free memory
            del model
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            continue

        # Compute score : mean r2
        r2_list = []
        for i, target in enumerate(targets):
            y_val_kinase = y_val[target].dropna()
            y_pred_kinase = y_pred.loc[y_val_kinase.index, target]
            try :
                r2_list.append(r2_score(y_val_kinase, y_pred_kinase))
            except:
                print('Warning: r2_score failed for target', target)
                pass
        score = np.median(r2_list)

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
            joblib.dump(best_model, model_path + '/model.joblib')

        print('Score:', score)
        print(f'Best score: {best_score:.4f} with params {best_params}')

        # Free memory
        del model
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    joblib.dump(best_model, model_path + '/model.joblib')
    print('Best model saved to', f'{model_path}/model.joblib')
    print('Best score:', best_score)
    print('Best params:', best_params)    

def predict_PB_MT(data_path, model_path : str, preds_path : str = None):

    """
    Predicts the targets for a dataset using Random Forests
    
    Parameters
    ----------
    data_path : str
        Path to the data
    model_path : str
        Path to the models
    preds_path : str
        Path to save the predictions
    
    Returns
    -------
    preds : pd.DataFrame
        Predictions
    """

    data = pd.read_csv(data_path)
    preds = data.copy()

    fps = compute_fps(data).astype(float).values
    targets = data.drop(['SMILES', 'InChIKey', 'Split', 'Subset', 'MinInterSetTd', 'set_index'], axis=1, errors='ignore').columns.tolist()

    model = joblib.load(model_path + '/model.joblib')
    predictions = model.predict(fps)
    for i, target in enumerate(targets):
        preds[target] = predictions[:, i]

    if preds_path is not None:
        os.makedirs(os.path.dirname(preds_path), exist_ok=True)
        preds.to_csv(preds_path, index=False)

    return preds