
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings('ignore')

from src.colors import *

# plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})

lfs = 20
mfs = 18
sfs = 16

def dataset_stats(df):
    df = df.drop(['SMILES', 'Subset', 'MinInterSetTd', 'InChIKey'], axis=1)
    print(f'Kinases: {len(df.columns)}')
    print(f'Molecules: {len(df)}')
    print(f'Datapoints: {df.count().sum()} ')
    print(f'Density: {df.count().sum() / (len(df.columns)* len(df))*100:.2f}')


def calculate_balance(df):
    targets = df.drop(['SMILES', 'Subset', 'InChIKey', 'MinInterSetTd'], axis=1).columns.tolist()
    train, valid, test = [], [], []
    for i, t in enumerate(targets):
        ntrain = len(df[df.Subset == 'train'][t].dropna())
        nvalid = len(df[df.Subset == 'valid'][t].dropna())
        ntest = len(df[df.Subset == 'test'][t].dropna())
        ntot = ntrain + nvalid + ntest
        train.append(ntrain / ntot)
        valid.append(nvalid / ntot)
        test.append(ntest / ntot)

    return pd.DataFrame({'train':train, 'valid':valid, 'test':test})


def dataset_balance(df):
    balance = calculate_balance(df)
    for t in ['train', 'valid', 'test']:
        # Print mean and std
        print(f'{t}: {np.mean(balance[t]*100):.1f} +/- {np.std(balance[t]*100):.1f}')

    return balance.melt(var_name='Subset', value_name='Balance')

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_dataset_distribution(data_balance, data_activity, data_tanimoto, fname=None):

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    # sns.despine(offset=10, trim=False)

    # A. Balance
    data = data_balance
    for subset, size in zip(['train', 'valid', 'test'], [0.8, 0.1, 0.1]):
        data.loc[data.Subset == subset, 'Balance'] -= size

    ax[0]= sns.violinplot(x='Split', y='Balance', data=data, hue='Subset', palette=my_light_colors, saturation=1, ax=ax[0])
    ax[0].set_title('A. Balance', size=lfs)
    ax[0].set_ylim(-0.1, 0.1)
    # ax[0].set_yticks([-0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08])
    ax[0].set_yticks([ -0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075])

    # Change legend labels to Train, Validation, Test
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles=handles, labels=['Train', 'Validation', 'Test'], 
            loc='center left', bbox_to_anchor=(0,0.15), frameon=False, fontsize=sfs)

    # B. Activity
    data = data_activity
    ax[1] = sns.violinplot(x='Split', y='pActivity', data=data, hue='Subset', palette=my_light_colors, saturation=1, ax=ax[1])
    ax[1].set_title('B. pActivity', size=lfs)
    ax[1].legend([],[], frameon=False)
    ax[1].set_ylim(2, 14)
    ax[1].set_yticks([4, 6, 8, 10, 12])

    # C. MinInterset Tanimoto distance
    data = data_tanimoto
    ax[2] = sns.violinplot(x='Split', y='MinInterSetTd', data=data, hue='Subset', palette=my_light_colors, ax=ax[2])
    # ax[2] = sns.boxenplot(x='Split', y='MinInterSetTd', data=data, hue='Subset',  ax=ax[2], saturation=1, width=0.1)
    ax[2].set_title('C. Min. interset Tan. dist.', size=lfs)
    ax[2].legend([],[], frameon=False)
    ax[2].set_ylim(0, 1)
    ax[2].set_yticks([0.2, 0.4, 0.6, 0.8])

    for i in range(3):
        ax[i].grid(linestyle='--', alpha=0.5, axis='y')
        ax[i].tick_params(axis='x', which='major', labelsize=mfs)
        ax[i].tick_params(axis='y', which='major', labelsize=sfs)
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        for loc in ['top', 'bottom' ,'left', 'right']: ax[i].spines[loc].set_color('k')
        for loc in ['top', 'right']: ax[i].spines[loc].set_color('w')

        # Set box and whisker colors
        childrens = ax[i].get_children()
        for k in [2,3,16,17]:
            childrens[k].set_color(leiden)
        for k in [7,8,20,21]:
            childrens[k].set_color(science)
        for k in [12,13,24,25]:
            childrens[k].set_color(galapagos)

        # Set edge colors
        for j, k  in enumerate([0,2,4,6,8,10]):
            ax[i].collections[k].set_edgecolor(my_colors[j % 3])
            

    if fname is not None:
        fig.savefig(fname, dpi=600, bbox_inches='tight')

def compute_stats(df_true : pd.DataFrame , df_pred : pd.DataFrame):

    """ Compute R2 and RMSE for each target in df_true and df_pred """
    
    df_stats = pd.DataFrame()
    nmols = len(df_true)
    targets = df_true.drop('SMILES', axis=1).columns
    for t in targets:
        true = df_true[t].dropna()
        idx = true.index
        pred = df_pred.loc[idx, t]
        df_stats = pd.concat([df_stats, pd.DataFrame({'Target' : [t]*2, 'Density': len(true)/nmols*100, 'Metric' : ['R2', 'RMSE'], 'Value' : [r2_score(true, pred), np.sqrt(mean_squared_error(true, pred))]})], ignore_index=True)

    for m in ['R2', 'RMSE']:    
        x = df_stats[df_stats.Metric == m].Value.to_numpy()
    
    return df_stats

def get_metrics_per_target_for_all_models():

    df = pd.DataFrame(columns=['Target', 'Split', 'Model', 'Mode', 'Dataset', 'Task', 'Metric', 'Value'])

    for model in ['RF', 'XGB', 'PB', 'CP', 'CP_ImputedMean', 'CP_ImputedRF', 'pQSAR']:
        for task in ['ST', 'MT']:
            if task == 'ST':
                if 'Imputed' in model or model in ['pQSAR', 'PB']: continue
            else:
                if model in ['RF', 'XGB']: continue
            
            for dataset in ['kinase1000', 'kinase200']:
                for split in ['RGES', 'DGBC']:
                    true = pd.read_csv(f'ModelInputs/{dataset}/{split}/Original/test.csv')
                    
                    for mode in ['Default', 'HyperOpt', 'DataLeakage']:
                        if mode == 'DataLeakage' and model != 'pQSAR': continue
                        elif mode == 'HyperOpt' and model == 'pQSAR': continue
                        elif mode == 'HyperOpt' and dataset == 'kinase1000' and model not in ['PB', 'CP']: continue

                        try :
                            preds = pd.read_csv(f'Predictions/{model}/{task}/{dataset}/{split}/{mode}/predictions.csv')
                        except:
                            print(f'Pass - no Predictions/{model}/{task}/{dataset}/{split}/{mode}/predictions.csv.gz file')
                            continue

                        metrics = compute_stats(true, preds)
                        metrics['Model'], metrics['Task'], metrics['Dataset'], metrics['Split'], metrics['Mode'] = model, task, dataset, split, mode
                        if model == 'CP_ImputedMean': metrics['Model'] = 'CPimputedMean'
                        if model == 'CP_ImputedRF': metrics['Model'] = 'CPimputedRF'
                        
                        df = pd.concat([df, metrics], ignore_index=True) 

    return add_model_names(df)

def add_model_names(df):
    """ Add model name to the dataframe """
    df.loc[(df.Model == 'RF') & (df.Mode == 'Default'), 'Name'] = 'RF/ST'
    df.loc[(df.Model == 'RF') & (df.Mode == 'HyperOpt'), 'Name'] = 'RF/ST/opt'

    df.loc[(df.Model == 'XGB') & (df.Mode == 'Default'), 'Name'] = 'XGB/ST'
    df.loc[(df.Model == 'XGB') & (df.Mode == 'HyperOpt'), 'Name'] = 'XGB/ST/opt'

    df.loc[(df.Model == 'PB') & (df.Mode == 'Default'), 'Name'] = 'PB/MT'
    df.loc[(df.Model == 'PB') & (df.Mode == 'HyperOpt'), 'Name'] = 'PB/MT/opt'

    df.loc[(df.Model == 'CP') & (df.Task == 'ST') & (df.Mode == 'Default'), 'Name'] = 'CP/ST'
    df.loc[(df.Model == 'CP') & (df.Task == 'ST') & (df.Mode == 'HyperOpt'), 'Name'] = 'CP/ST/opt'

    df.loc[(df.Model == 'CP') & (df.Task == 'MT') & (df.Mode == 'Default'), 'Name'] = 'CP/MT'
    df.loc[(df.Model == 'CP') & (df.Task == 'MT') & (df.Mode == 'HyperOpt'), 'Name'] = 'CP/MT/opt'

    df.loc[(df.Model == 'CPimputedMean') & (df.Task == 'MT') & (df.Mode == 'Default'), 'Name'] = 'CP iMean/MT'
    df.loc[(df.Model == 'CPimputedMean') & (df.Task == 'MT') & (df.Mode == 'HyperOpt'), 'Name'] = 'CP iMean/MT/opt'

    df.loc[(df.Model == 'CPimputedRF') & (df.Task == 'MT') & (df.Mode == 'Default'), 'Name'] = 'CP iRF/MT'
    df.loc[(df.Model == 'CPimputedRF') & (df.Task == 'MT') & (df.Mode == 'HyperOpt'), 'Name'] = 'CP iRF/MT/opt'

    df.loc[(df.Model == 'pQSAR') & (df.Task == 'MT') & (df.Mode == 'Default'), 'Name'] = 'pQSAR'
    df.loc[(df.Model == 'pQSAR') & (df.Task == 'MT') & (df.Mode == 'DataLeakage'), 'Name'] = 'pQSAR/dl'

    return df

def compute_statistics_of_metrics(df):

    """
    Compute statistics of metrics for all models and datasets
    """

    df.dropna(inplace=True)

    df_stats = pd.DataFrame()
    for model in ['RF', 'XGB', 'CP', 'CPimputedMean', 'CPimputedRF', 'pQSAR', 'PB']:
        for dataset in ['kinase200', 'kinase1000']:
            for split in ['RGES', 'DGBC']:
                for task in ['ST', 'MT']:
                    for mode in ['Default', 'HyperOpt', 'DataLeakage']:
                        for metric in ['R2', 'RMSE']:
                            df_selected = df[(df.Model == model) & (df.Metric == metric) & (df.Dataset == dataset) & (df.Split == split) & (df.Task == task) & (df.Mode == mode)]
                            if df_selected.shape[0] == 0:
                                continue
                            x = df_selected.Value.to_numpy()
                            # print(model, df_selected.Name.unique())
                            name = df_selected.Name.unique()[0]
                            # print(name, x.shape, np.std(x))
                            df_tmp = pd.DataFrame({
                                'Name' : name,
                                'Model' : model,
                                'Dataset' : dataset, 
                                'Split' : split,
                                'Task' : task,
                                'Mode' : mode,
                                'Metric' : metric,   'Median' : np.median(x), 'Mean' : np.mean(x), 'Std' : np.std(x)}, index=[0])
                            df_stats = pd.concat([df_stats, df_tmp], ignore_index=True)

    return df_stats.dropna().reset_index(drop=True)

def compute_effect_of_hyperparam_opt(df_stats):

    df = df_stats[(df_stats.Dataset == 'kinase200') & (df_stats.Model != 'pQSAR')]
    df.dropna(inplace=True)

    # Average difference between default and hyperopt of R2 median values for RGES split
    x = np.mean(df[(df.Mode == 'HyperOpt') & (df.Split == 'RGES') & (df.Metric == 'R2')].Median.values - df[(df.Mode == 'Default') & (df.Split == 'RGES') & (df.Metric == 'R2')].Median.values)
    print(f'Average difference between default and hyperopt of R2 median values for RGES split = {x:.2f}') 

    # Average difference between default and hyperopt of RMSE median values for RGES split
    x = np.mean(df[(df.Mode == 'HyperOpt') & (df.Split == 'RGES') & (df.Metric == 'RMSE')].Median.values - df[(df.Mode == 'Default') & (df.Split == 'RGES') & (df.Metric == 'RMSE')].Median.values)
    print(f'Average difference between default and hyperopt of RMSE median values for RGES split = {x:.2f}')

    # Average difference between default and hyperopt of R2 median values for DGBC split
    x = np.mean(df[(df.Mode == 'HyperOpt') & (df.Split == 'DGBC') & (df.Metric == 'R2')].Median.values - df[(df.Mode == 'Default') & (df.Split == 'DGBC') & (df.Metric == 'R2')].Median.values)
    print(f'Average difference between default and hyperopt of R2 median values for DGBC split = {x:.2f}')

    # Average difference between default and hyperopt of RMSE median values for DGBC split
    x = np.mean(df[(df.Mode == 'HyperOpt') & (df.Split == 'DGBC') & (df.Metric == 'RMSE')].Median.values - df[(df.Mode == 'Default') & (df.Split == 'DGBC') & (df.Metric == 'RMSE')].Median.values)
    print(f'Average difference between default and hyperopt of RMSE median values for DGBC split = {x:.2f}')

    diff = pd.DataFrame()

    for split in df.Split.unique():
        for model in df.Model.unique():
            for task in df.Task.unique():
                for metric in df.Metric.unique():
                    data = df[(df.Split == split) & (df.Model == model) & (df.Task == task) & (df.Metric == metric)]
                    default = data[data.Mode == 'Default'].Median.values
                    opt = data[data.Mode == 'HyperOpt'].Median.values
                    x = opt - default
                    df_ = pd.DataFrame({'Model' : model, 'Split' : split, 'Task' : task, 'Metric' : metric, 'Diff' : opt-default})
                    diff = pd.concat([diff, df_], ignore_index=True)
    return diff

def format_boxes(ax):

    box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
    num_patches = len(box_patches)
    lines_per_boxplot = len(ax.lines) // num_patches

    for i, patch in enumerate(box_patches):
        col = leiden if i % 2 == 0 else science
        col_light = leiden_light if i % 2 == 0 else science_light
        patch.set_facecolor(col_light)
        patch.set_edgecolor(col)
        patch.set_alpha(1.0)
        patch.set_alpha(1.0)

    # Loop over them here, and use the same color as above
        for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
            line.set_color(col)
            line.set_mfc('w')  # facecolor of fliers
            line.set_mec(col)  # edgecolor of fliers


def performance_plot(df: pd.DataFrame, xtickslabels : str = None, fname : str = None,
    x='Name', y='Value', row='Metric', hue='Split'):
    
    """Plot performance metrics for a given dataset"""

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    fig.figure.subplots_adjust(wspace=0.1, hspace=0.1)


    # R2
    ax[0] = sns.boxplot(data=df[df.Metric=='R2'], x=x, y=y, hue=hue, ax=ax[0], 
        palette=my_light_colors, saturation=1.0, 
    )
    ax[0].set_ylabel(r'$R^2$', size=lfs)
    ax[0].set_ylim(-1.,1)
    ax[0].set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    ax[0].set_axisbelow(True)
    ax[0].set_xticklabels([])
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.8, 1.1), ncol=2, fontsize=mfs, frameon=False)
    format_boxes(ax[0])

    # RMSE
    ax[1] = sns.boxplot(data=df[df.Metric=='RMSE'], x=x, y=y, hue=hue, ax=ax[1],
        palette=my_light_colors, saturation=1.0,    
    )
    ax[1].set_ylabel(r'$\mathrm{RMSE}$', size=lfs)
    ax[1].set_ylim(0.1,1.5)
    ax[1].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    # Remove legend
    ax[1].legend_.remove()


    if xtickslabels is not None:
        ax[1].set_xticklabels(xtickslabels, rotation=90, size=lfs)
    else:
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, size=lfs)
    format_boxes(ax[1])

    # General
    fig.align_ylabels(ax)
    for a in ax:
        for loc in ['top', 'bottom' ,'left', 'right']: a.spines[loc].set_color('k')
        for loc in ['top', 'right']: a.spines[loc].set_color('w')
        a.set_title('')
        a.grid(axis='y', alpha=0.5, c='gray', linestyle='--')
        a.set_xlabel('')
        a.tick_params(axis='y', labelsize=sfs)

    # Save
    if fname is not None:
        fig.savefig(fname, dpi=600, bbox_inches='tight')


def plot_effect_density(df, fname=None):

    fig, ax = plt.subplots(2,1, figsize=(8,8), sharex=True)
    fig.subplots_adjust(hspace=0.1)

    for i, metric in enumerate(['R2', 'RMSE']):

        tmp = df[df.Metric == metric]

        # Spearman correlation
        rhor, pval = spearmanr(tmp[tmp.Split == 'RGES'].Density, tmp[tmp.Split == 'RGES'].Value)
        rhobc, pval = spearmanr(tmp[tmp.Split == 'DGBC'].Density, tmp[tmp.Split == 'DGBC'].Value)

        sns.regplot(x='Density', y='Value', data=tmp[tmp.Split == 'RGES'], ax=ax[i], label=fr'RGES - $\rho$: {rhor:.2f}', color=leiden, scatter_kws={'s': 50, 'alpha': 0.5, 'marker':'t'}, line_kws={'lw': 2, 'ls': ':'}, ci=None)
        sns.regplot(x='Density', y='Value', data=tmp[tmp.Split == 'DGBC'], ax=ax[i], label=rf'DGBC - $\rho$: {rhobc:.2f}', color=science, scatter_kws={'s': 50, 'alpha': 0.5, 'marker': 's'}, line_kws={'lw': 2, 'ls': ':'}, ci=None)

        ax[i].legend(framealpha=0.5, fontsize=mfs, ncol=1, edgecolor='w', markerscale=2)
        ax[i].set_xlim(0.01, 9.99)

    ax[0].set_ylim(-1.49,0.99)
    ax[0].set_ylabel(r'$R^2$', size=lfs)
    ax[0].set_xlabel('')
    # ax[0].text(-0.15, 0.9, 'A', transform=ax[0].transAxes, size=25, weight='bold')    

    ax[1].set_ylim(0.01,1.99)
    ax[1].set_ylabel(r'$\mathrm{RMSE}$', size=lfs)
    ax[1].set_xlabel(r'$\mathrm{Density\, (\%)}$', size=lfs)
    # ax[1].text(-0.15, 0.9, 'B', transform=ax[1].transAxes, size=25, weight='bold')    

    fig.align_ylabels(ax)
    for a in ax:
        for loc in ['top', 'bottom' ,'left', 'right']: a.spines[loc].set_color('k')
        for loc in ['top', 'right']: a.spines[loc].set_color('w')
        a.grid(alpha=0.5, c='gray', linestyle='--')
        a.tick_params(axis='y', labelsize=sfs)
        a.tick_params(axis='x', labelsize=sfs)

    if fname is not None:
        fig.savefig(fname, dpi=600, bbox_inches='tight')

def plot_effect_datapoints(df, fname = None):


    r2_r_200 = df[(df.Metric == 'R2') & (df.Dataset == 'kinase200') & (df.Split == 'RGES') ].Value.values
    r2_r_1000  = df[(df.Metric == 'R2') & (df.Dataset == 'kinase1000') & (df.Split == 'RGES') ].Value.values
    diff_r2_r = r2_r_1000 - r2_r_200

    r2_bc_200 = df[(df.Metric == 'R2') & (df.Dataset == 'kinase200') & (df.Split == 'DGBC') ].Value.values
    r2_bc_1000  = df[(df.Metric == 'R2') & (df.Dataset == 'kinase1000') & (df.Split == 'DGBC') ].Value.values
    diff_r2_bc = r2_bc_1000 - r2_bc_200

    rmse_r_200 = df[(df.Metric == 'RMSE') & (df.Dataset == 'kinase200') & (df.Split == 'RGES') ].Value.values
    rmse_r_1000  = df[(df.Metric == 'RMSE') & (df.Dataset == 'kinase1000') & (df.Split == 'RGES') ].Value.values
    diff_rmse_r = rmse_r_1000 - rmse_r_200

    rmse_bc_200 = df[(df.Metric == 'RMSE') & (df.Dataset == 'kinase200') & (df.Split == 'DGBC') ].Value.values
    rmse_bc_1000  = df[(df.Metric == 'RMSE') & (df.Dataset == 'kinase1000') & (df.Split == 'DGBC') ].Value.values
    diff_rmse_bc = rmse_bc_1000 - rmse_bc_200

    fig, ax = plt.subplots(2,2, figsize=(8,4), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.set(facecolor='white')

    bins1= np.arange(-1,0.05,0.05)
    bins2 = np.arange(0,1.0,0.05)

    ax[0,0].hist(diff_r2_r, bins=bins1, alpha=1.0, color=galapagos_light, histtype='stepfilled', edgecolor=galapagos, lw=2, hatch='//', density=True)
    ax[0,0].hist(diff_r2_r, bins=bins2, alpha=1.0, color=purple_light, histtype='stepfilled', edgecolor=purple, lw=2, hatch="o", density=True)

    ax[1,0].hist(diff_r2_bc, bins=bins1, alpha=1.0, color=galapagos_light, histtype='stepfilled', edgecolor=galapagos, lw=2, hatch='//', density=True)
    ax[1,0].hist(diff_r2_bc, bins=bins2, alpha=1.0, color=purple_light, histtype='stepfilled', edgecolor=purple, lw=2, hatch='o', density=True)

    ax[0,1].hist(diff_rmse_r, bins=bins1, alpha=1.0, color=purple_light, histtype='stepfilled', edgecolor=purple, lw=2, hatch='o', density=True, label='Worsened')
    ax[0,1].hist(diff_rmse_r, bins=bins2, alpha=1.0, color=galapagos_light, histtype='stepfilled', edgecolor=galapagos, lw=2, hatch='//', density=True, label='Improved')

    ax[1,1].hist(diff_rmse_bc, bins=bins1, alpha=1.0, color=purple_light, histtype='stepfilled', edgecolor=purple, lw=2, hatch='o', density=True)
    ax[1,1].hist(diff_rmse_bc, bins=bins2, alpha=1.0, color=galapagos_light, histtype='stepfilled', edgecolor=galapagos, lw=2, hatch='//', density=True)

    ax[0,1].legend(framealpha=0.2, fontsize=mfs, loc='upper right', ncol=2,  frameon=False,bbox_to_anchor=(1.08, 1.4))
    ax[1,0].set_xlabel(r'$\Delta R^2$', size=lfs)
    ax[1,1].set_xlabel(r'$\Delta \mathrm{RMSE}$', size=lfs)

    ax[0,0].set_xlim(-0.75,.75)
    ax[0,0].set_yticks([])
    ax[0,0].set_yticklabels([])

    ax[0,0].set_ylabel('RGES split', size=lfs)
    ax[1,0].set_ylabel('DGBC split', size=lfs)

    for a in ax.flatten():
        a.xaxis.set_minor_locator(MultipleLocator(0.1))
        a.grid(which='minor', linestyle='--')
        a.grid(which='major', linestyle='-')
        a.set_axisbelow(True)
        a.tick_params(axis='both', which='major', labelsize=sfs)
        for loc in ['top', 'bottom' ,'left', 'right']: a.spines[loc].set_color('k')
        for loc in ['top', 'right']: a.spines[loc].set_color('w')

    if fname is not None:
        fig.savefig(fname, dpi=600, bbox_inches='tight')

def plot_pQSAR_performance(df, fname=None):

    df = df[(df.Dataset == 'kinase200') & (df.Name.isin(['RF/ST', 'pQSAR/dl', 'pQSAR']))]
    names = [r'$\mathrm{RF}$', r'$\mathrm{pQSAR_{w/\;test}}$', r'$\mathrm{pQSAR_{w/o\;test}}$']

    fig = sns.catplot(data=df, kind='box', x='Split', y='Value', row='Metric', hue='Name', 
        palette=my_light_colors, saturation=1.0, sharey=False, aspect=2, height=4, 
        legend_out=False, hue_order=['RF/ST', 'pQSAR/dl', 'pQSAR']
    )

    sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.97, 1.0), bbox_transform=plt.gcf().transFigure, ncol=3, frameon=False, title=None, fontsize=sfs)
    for t, l in zip(fig._legend.texts, names):
        t.set_text(l)

    fig.figure.subplots_adjust(wspace=0.1, hspace=0.1)
    # make the background white
    fig.set(facecolor='white')

    for row_val, ax in fig.axes_dict.items():
        ax.grid(axis='y', alpha=0.5, c='gray')
        for loc in ['top', 'bottom' ,'left', 'right']: ax.spines[loc].set_color('k') 
        
        if row_val == 'R2':
            ax.set_title('')
            ax.set_ylabel(r'$R^2$', size=lfs)
            ax.set_ylim(-0.5,1)
            # ax.set_yticks([-0.4, 1.2, 0.2])
            # ax.set_yticklabels(['0.0', '1.0', '0.2']
        if row_val == 'RMSE':
            ax.set_title('')
            ax.set_ylabel(r'$\mathrm{RMSE}$', size=lfs)
            ax.set_ylim(0,1.5)       
            ax.set_xlabel('')#Model', size=lfs)

        ax.grid(axis='y', alpha=0.5, c='gray', linestyle='--')
        ax.tick_params(axis='y', which='major', labelsize=sfs)
        ax.tick_params(axis='x', which='major', labelsize=mfs)

        box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
        num_patches = len(box_patches)
        lines_per_boxplot = len(ax.lines) // num_patches

        for i, patch in enumerate(box_patches):
            if i % 3 == 0:
                col = my_colors[0]
                col_light = my_light_colors[0]
            elif i % 3 == 1:
                col = my_colors[1]
                col_light = my_light_colors[1]
            elif i % 3 == 2:
                col = my_colors[2]
                col_light = my_light_colors[2]
            patch.set_facecolor(col_light)
            patch.set_edgecolor(col)
            patch.set_alpha(1.0)
            patch.set_alpha(1.0)

        # Loop over them here, and use the same color as above
            for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
                line.set_color(col)
                line.set_mfc('w')  # facecolor of fliers
                line.set_mec(col)  # edgecolor of fliers
        
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', dpi=600)