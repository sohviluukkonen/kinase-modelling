#General modules
import os
from tkinter import X
import pandas as pd
import joblib
import tqdm

from src.rf import train_RF_STs, predict_RF_STs
from src.utils import mkdirs

from sklearn.cross_decomposition import PLSRegression


#Function for training the PLS model
def train_pls(x, y):
    """
    Train PLS model with variable number of components and select the best model
    based on the R2 score and the number of components.
    
    Parameters
    ----------
    x : pd.DataFrame
        Dataframe with the features.
    y : pd.Series
        Series with the target values.
    """

    score_t = -1000000
    
    for components in tqdm.tqdm(range(2,26), desc='Training PLS models'):
        pls = PLSRegression(n_components=components)
        temp_model = pls.fit(x, y)
        score = temp_model.score(x, y) - 0.002*components 
        if score > score_t:
            score_t = score
            model = temp_model
    return model
        
def train_pQSAR(data_path : str, rf_path : str, model_path : str, subset : str = None):
    """
    Train pQSAR models for all targets in the dataset.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset.
    rf_path : str
        Path to the RF models.
    model_path : str
        Path to save the pQSAR models.
    subset : str, optional
        Subset of the dataset to train the models on, by default None
    """

    mkdirs(model_path)

    data = pd.read_csv(data_path)
    if subset:
        data = data[data.Subset == subset]
    targets = data.drop(['SMILES', 'Subset', 'MinInterSetTd'], errors='ignore', axis=1).columns.tolist()

    rf_preds = predict_RF_STs(data_path, rf_path)

    for target in tqdm.tqdm(targets, desc='Training pQSAR models'):
        y_train = data[target].dropna()
        x_train = rf_preds.drop(['SMILES',target], axis=1).loc[y_train.index]
        model = train_pls(x_train, y_train)
        joblib.dump(model, f'{model_path}/{target}.joblib')

def predict_pQSAR(data_path: str, rf_path : str , pls_path : str, preds_path : str = None, subset : str = None):

    """
    Predict pQSAR models for all targets in the dataset.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset.
    rf_path : str
        Path to the RF models.
    pls_path : str
        Path to the pQSAR models.
    preds_path : str, optional
        Path to save the predictions, by default None
    subset : str, optional
        Subset of the dataset used for predictions, by default None
    """
    
    data = pd.read_csv(data_path)
    if subset:
        data = data[data.Subset == subset]
    targets = data.drop(['SMILES', 'Subset', 'MinInterSetTd'], errors='ignore', axis=1).columns.tolist()

    rf_preds = predict_RF_STs(data_path, rf_path)

    preds = data.copy()

    for target in tqdm.tqdm(targets, desc='Predicting with pQSAR models'):
        y_test = data[target].dropna()
        x_test = rf_preds.drop(['SMILES',target], axis=1).loc[y_test.index]

        model = joblib.load(f'{pls_path}/{target}.joblib')
        preds.loc[y_test.index, target] = model.predict(x_test)

    if preds_path:
        mkdirs(preds_path)
        preds.to_csv(f'{preds_path}/predictions.csv', index=False)
    
    return preds

def pQSAR_model_validation():

    """ Validation of pQSAR2.0 model implementation by reproducing results from Martin et al. """
    
    dataset = 'Assays'
    for split in  ['Random', 'Realistic']:
        
        data_path = f'ModelInputs/{dataset}/{split}'
        rf_path = f'Models/RF/ST/{dataset}/{split}'
        rf_preds_path = f'Predictions/RF/ST/{dataset}/{split}'
        pls_path = f'Models/pQSAR/MT/{dataset}/{split}'
        pls_preds_path = f'Predictions/pQSAR/MT/{dataset}/{split}'

        train_RF_STs(f'{data_path}/exp_train.csv', rf_path)
        predict_RF_STs(f'{data_path}/exp_test.csv', rf_path, rf_preds_path)  
        train_pQSAR(f'{data_path}/exp_all.csv', rf_path, pls_path )
        predict_pQSAR(f'{data_path}/exp_test.csv', rf_path, pls_path, pls_preds_path)


def run_pQSAR_model(dataset : str, split : str, data_leakage : str = 'Default'):

    """
    Train and make predictions with pQSAR model.
    
    Parameters
    ----------
    dataset : str
        Dataset to run the model on.
    split : str
        Split to run the model on.
    data_leakage : str, optional
        Data leakage strategy, by default 'Default'
    """
    
    data_path = f'ModelInputs/{dataset}/{split}/Original'
    rf_path = f'Models/RF/ST/{dataset}/{split}/Default'
    pls_path = f'Models/pQSAR/MT/{dataset}/{split}/{data_leakage}'
    pls_preds_path = f'Predictions/pQSAR/MT/{dataset}/{split}/{data_leakage}'

    if data_leakage == 'Default':
        train_pQSAR(f'{data_path}/train.csv', rf_path, pls_path )
    else:
        train_pQSAR(f'{data_path}/all.csv', rf_path, pls_path )

    predict_pQSAR(f'{data_path}/test.csv', rf_path, pls_path, pls_preds_path)



