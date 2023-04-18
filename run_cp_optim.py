import os
import sys
import optuna

from src.chemprop import train_chemprop_MT

def fetch_val_r2(log):
    """
    Fetch validation R2 from Chemprop log file.

    Parameters
    ----------
    log : str
        Path to Chemprop log file.

    Returns
    -------
    rmse_val : float
        Validation RMSE.
    """
    with open(log) as myfile:
        for row in myfile:
            if 'Ensemble test r2' in row:
                row = row.split(" ")
                idx = row.index('=')
                idx = idx+1
                r2_val = row[idx]
    myfile.close()
    return float(r2_val)


def objective(trial):
    """
    Objective function for Optuna.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Trial object.
    
    Returns
    -------
    rmse_val : float
        Validation RMSE.
    """
    
    i = trial.number

    # Define hyperparameters
    size = trial.suggest_int("size", 600, 2000, 100)
    depth = trial.suggest_int("depth", 3, 4, 1)
    dropout = trial.suggest_float("dropout", 0, 0.2)
    ffn_num_layers = trial.suggest_int("ffn_num_layers", 2, 4, 1)
    activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "SELU", "PReLU", "tanh", "ELU"]) #"Hardswish","Mish","CELU"])
    max_lr = trial.suggest_categorical("max_lr", [0.0005, 0.001, 0.0025, 0.005])
    epochs = trial.suggest_categorical("epochs", [30, 40, 50])
    bias = trial.suggest_categorical("bias", [True, False])

    kwargs = {
        'hidden_size' : size, 
        'depth' : depth,
        'dropout' : dropout,
        'ffn_num_layers' : ffn_num_layers,
        'activation' : activation,
        'max_lr' : max_lr,
        'epochs' : epochs
    }

    if bias : kwargs['bias'] = ' '

    # Define targets used for hyperparameter optimization
    targets = ['P00533_WT', 'P04626_WT', 'P06239_WT', 'Q5S007_WT', 'O75116_WT']
    kwargs['target_columns'] = ' '.join(targets)

    kwargs['gpu'] = 2

    print(kwargs)

    data_path = 'ModelInputs/kinase1000/DGBC/Original/train.csv'
    valid_path = 'ModelInputs/kinase1000/DGBC/Original/valid.csv'
    test_path = 'ModelInputs/kinase1000/DGBC/Original/valid.csv'
    model_path = f'ChempropHyperOpt/params_{i:03d}'
    
    train_chemprop_MT(data_path, valid_path, test_path, model_path, **kwargs)
    log = f'ChempropHyperOpt/params_000/quiet.log'
    metric = fetch_val_r2(log)
    os.system(f'rm -rf {model_path}/fold_0')

    return metric

if __name__ == '__main__':

    """
    Run hyperparameter optimization of chemprop multi-taks model using Optuna.
    """

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)
    print(study.best_params)
