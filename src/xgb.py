import os
import tqdm.auto as tqdm
import joblib
import pandas as pd

from .utils import mkdirs, compute_fps

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, r2_score

def XGB_hyperparams():
    """
    Returns a dictionary of hyperparameters for XGBoost
    """
    return {"max_depth": [6,7,8,9,10,12,16],
                "learning_rate": [0.01,0.02,0.03,0.04,0.05],
                "min_child_weight":[1,2,4,8,16,32],
                "n_estimators": [50, 100, 200, 500, 750, 1000, 1500,2000],
                "colsample_bytree": [0.5,0.6,0.7,0.8],
                "subsample": [0.4,0.5,0.6,0.7,0.8,0.9,1],
                "scale_pos_weight": [10,25,30,50,60,80,100]
            }


def train_optim_XGB_STs(
    data_path,
    val_path, 
    model_path : str, 
    seed : int = 2022,
    n_jobs : int = 16,
    n_iter : int = 100):
    """
    Trains a XGBoost model for each target using Randomised Search CV
    
    Parameters
    ----------
    data_path : str
        Path to the training data
    val_path : str
        Path to the validation data
    model_path : str
        Path to save the models
    seed : int
        Random seed
    n_jobs : int
        Number of jobs to run in parallel
    n_iter : int
        Number of iterations for Randomised Search CV
    """

    mkdirs(model_path)

    data = pd.read_csv(data_path)
    data['split_index'] = -1        

    val = pd.read_csv(val_path)
    val['split_index'] = 0

    data = pd.concat([data, val], axis=0, ignore_index=True)
    targets = data.drop(['SMILES', 'InChIKey', 'Split', 'Subset', 'MinInterSetTd', 'split_index'], axis=1, errors='ignore').columns.tolist()

    fps = compute_fps(data)

    for target in tqdm.tqdm(targets, desc='Training XGB models'):

        if os.path.exists(f'{model_path}/{target}.joblib'):
            continue
        
        y = data[target].dropna()
        X = fps.iloc[y.index]
        splits = PredefinedSplit(test_fold = data.iloc[y.index].split_index)

        model = xgb.XGBRegressor(random_state = seed, gpu_id=0, tree_method='gpu_hist')
        rdnSearch = RandomizedSearchCV(model, XGB_hyperparams(), n_iter=n_iter, cv=splits, n_jobs=1, scoring=make_scorer(r2_score), random_state=seed)
        rdnSearch.fit(X, y)
        joblib.dump(rdnSearch.best_estimator_, f'{model_path}/{target}.joblib')
        print('Best parameters for', target, rdnSearch.best_params_)
        print('Best score for', target, rdnSearch.best_score_)
        print('Best model saved to', f'{model_path}/{target}.joblib')    


def train_XGB_STs(data_path, model_path : str,  seed : int = 2022, gpu_id : int =  0):

    """
    Trains a XGBoost model for each target
    
    Parameters
    ----------
    data_path : str
        Path to the training data
    model_path : str
        Path to save the models
    seed : int
        Random seed
    gpu_id : int
        GPU ID to use
    """

    mkdirs(model_path)

    data = pd.read_csv(data_path)

    targets = data.drop('SMILES', axis=1).columns.tolist()

    fps = compute_fps(data)

    for target in tqdm.tqdm(targets, desc='Training XGBoosts'):
        y_train = data[target].dropna()
        X_train = fps.iloc[y_train.index]
        
        model = xgb.XGBRegressor(tree_method='gpu_hist', random_state = seed, gpu_id=gpu_id)
        model.fit(X_train, y_train) 
        joblib.dump(model, f'{model_path}/{target}.joblib')

def predict_XGB_STs(data_path, model_path : str, preds_path : str = None):
    """
    Predicts the targets for a dataset using XGBoost models
    
    Parameters
    ----------
    data_path : str
        Path to the data
    model_path : str
        Path to the models
    preds_path : str    
        Path to save the predictions
    """

    data = pd.read_csv(data_path)
    fps = compute_fps(data)
    targets = data.drop(['SMILES', 'InChIKey', 'Split', 'Subset', 'MinInterSetTd', 'set_index'], axis=1, errors='ignore').columns.tolist()

    preds = data.copy()

    for target in tqdm.tqdm(targets, desc='Predicting with XgBoosts'):
        model = joblib.load(f'{model_path}/{target}.joblib')
        preds.loc[:, target] = model.predict(fps)

    if preds_path:
        mkdirs(os.path.dirname(preds_path))
        preds.to_csv(preds_path, index=False)

    return preds