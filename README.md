# Large-scale modelling of sparse kinase activity data 

This repository includes the data and the scripts to create the large-scale kinase dataset and the accompaning modelling presented in [Large-scale modelling of sparse kinase activity data](link_to_preprint/paper).

## Data

### New large-scale kinase benchmark sets with two balanced splits without data leakage

The kinases sets were obtained with `dataset_creation.py`
1. retriving kinase data from Papyrus
2. filtering out kinases with less than 200 (kinase200) or 1000 (kinase1000) data points
3. two balanced splits without data leakage (random- and cluster-based) were applied to both sets
4. creates train, validation, test set input files for model training and evaluation

The four kinase benchmark datasets constructed in this work are at `data/datasets`.

### pQSAR datasets

To validate our implementation of the pQSAR2.0 model from Martin, et al.'s [Profile-QSAR 2.0: Kinase Virtual Screening Accuracy Comparable to Four-Concentration IC50s for Realistically Novel Compounds](http://dx.doi.org/10.1021/acs.jcim.7b00166), we retrived their datasets from the SI (`data/pQSAR`).

## Modelling

### Training and testing

The training and testing of all models (except pQSAR) for all dataset and splits were run with `run_models.py`. The pQSAR models were separetaly run with `run_pQSAR.py`

### Hyperparameter optimisation

The hyperparameter optimisation of the Random Forest and Xgboost models is done on the fly in `run_models.py`. For the D-MPNN models, it was done separately once with `run_cp_opt.py`.

## Analysis

The `Analysis.ipynb` notebook contains all steps to reproduce figures and analysis from the paper.

