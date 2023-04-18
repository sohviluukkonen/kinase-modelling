import os
import tqdm
import joblib
import pandas as pd

from .utils import mkdirs, compute_fps

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, r2_score


def RF_hyperparams():

    """
    Returns a dictionary of hyperparameters for Random Forests
    """
    
    return {
        'n_estimators': [50, 100, 200, 500, 750, 1000, 1500,2000],
        'max_depth': [None, 6,7,8,9,10,12,16],
        'min_samples_split': [2, 5, 12],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [1.0, 'sqrt', 'log2'],
    }

def train_optim_RF_STs(
    data_path,
    val_path,
    model_path : str,
    seed : int = 2022,
    n_jobs : int =  16,
    n_iter : int = 100):

    """
    Trains a Random Forest model for each target using Randomised Search CV
    
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
    targets = data.drop(['SMILES', 'InChIKey', 'Split', 'Subset', 'MinInterSetTd', 'set_index'], axis=1, errors='ignore').columns.tolist()

    for target in tqdm.tqdm(targets, desc='Training Random Forests with optimisation'):
        y = data[target].dropna()
        X = compute_fps(data.iloc[y.index])
        splits = PredefinedSplit(test_fold = data.iloc[y.index].split_index)

        model = RandomForestRegressor(random_state = seed, n_jobs=n_jobs)
        rdnSearch = RandomizedSearchCV(model, RF_hyperparams(), n_iter=n_iter, cv=splits, n_jobs=1, scoring=make_scorer(r2_score), random_state=seed)
        rdnSearch.fit(X, y)
        joblib.dump(rdnSearch.best_estimator_, f'{model_path}/{target}.joblib')
        print('Best parameters for', target, rdnSearch.best_params_)
        print('Best score for', target, rdnSearch.best_score_)
        print('Best model saved to', f'{model_path}/{target}.joblib')

def train_RF_STs(
    data_path,
    model_path : str,
    seed : int = 2022, 
    n_jobs : int =  16):

    """
    Trains a Random Forest model for each target
    
    Parameters
    ----------
    data_path : str
        Path to the training data
    model_path : str
        Path to save the models
    seed : int
        Random seed
    n_jobs : int
        Number of jobs to run in parallel
    """

    mkdirs(model_path)

    data = pd.read_csv(data_path)
    
    targets = data.drop(['SMILES', 'InChIKey', 'Split', 'Subset', 'MinInterSetTd', 'set_index'], axis=1, errors='ignore').columns.tolist()

    fps = compute_fps(data)

    for target in tqdm.tqdm(targets, desc='Training Random Forests'):
        y_train = data[target].dropna()
        X_train = fps.iloc[y_train.index]

        model = RandomForestRegressor(random_state = seed, n_jobs=n_jobs)
        model.fit(X_train, y_train) 
        joblib.dump(model, f'{model_path}/{target}.joblib')

def predict_RF_STs(data_path, model_path : str, preds_path : str = None):

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

    fps = compute_fps(data)
    targets = data.drop(['SMILES', 'InChIKey', 'Split', 'Subset', 'MinInterSetTd', 'set_index'], axis=1, errors='ignore').columns.tolist()

    preds = data.copy()

    for target in tqdm.tqdm(targets, desc='Predicting with Random Forests'):
        model = joblib.load(f'{model_path}/{target}.joblib')
        preds.loc[:, target] = model.predict(fps)

    if preds_path:
        mkdirs(os.path.dirname(preds_path))
        preds.to_csv(preds_path, index=False)

    return preds