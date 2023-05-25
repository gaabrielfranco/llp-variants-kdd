import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
from scipy.stats import ttest_ind
from llp_learn.util import compute_proportions

def get_dataset_variant(dataset):
    if "naive" in dataset:
         return "Naive"
    elif "simple" in dataset:
        return "Simple"
    elif "intermediate" in dataset:
        return "Intermediate"
    elif "hard" in dataset:
        return "Hard"
    else:
        return "unknown"

def plot_winning_figure(winning_df, filename, plot_type=None):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('font', size=6)

    split_method_map = {
        "Split-\nBag\nBootstrap": "SP-BS",
        "Split-\nbag\nShuffle": "SP-SH",
        "Split-\nbag\nK-fold": "SP-KF",
    }

    if plot_type == "algorithms":
        fig, axes = plt.subplots(nrows=1, ncols=len(winning_df.index.levels[0]), sharey=True, gridspec_kw={"width_ratios": [2, 3, 3]}, figsize=(3.5, 2)) # width, height
    else:
        fig, axes = plt.subplots(nrows=1, ncols=len(winning_df.index.levels[0]), sharey=True, figsize=(3.5, 2)) # width, height
    if plot_type == "variants":
        it_array = list(winning_df.index.levels[0])
        it_array[2], it_array[3] = it_array[3], it_array[2]
        rotation_ticks = 45
    else:
        it_array = winning_df.index.levels[0]
        rotation_ticks = 0

    # Rename level 1 index
    winning_df.index = winning_df.index.set_levels([split_method_map[x] for x in winning_df.index.levels[1]], level=1)

    for i, row in enumerate(it_array):
        ax = axes[i]
        winning_df.loc[row, :].plot(ax=ax, kind='bar', width=.8, stacked=True, color=["tab:blue", "tab:red",  "tab:grey"], fontsize=6, legend=None)
        
        ax.set_xlabel(row, weight='bold')
        ax.set_axisbelow(True)

        for tick in ax.get_xticklabels():
            tick.set_rotation(rotation_ticks)

        if i == 0:
            ax.tick_params(axis=u'x', which=u'both', length=0)
            ax.set_ylabel("Fraction of experiments", fontsize=6)
        else:
            ax.tick_params(axis=u'both', which=u'both', length=0)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--plot_type', "-p", type=str, required=True, help="Plot to generate")
args = parser.parse_args()

dataset_title_map = {}
error_legend_map = {
    "error_bag_abs": "Abs loss",
     "error_accuracy_validation": "Oracle"
}

"""
Attention:
    This can change depending on the experiments - e.g. n_folds and validation perc size
"""
split_method_map = {
    'split-bag-bootstrap': "Split-\nBag\nBootstrap",
    'split-bag-shuffle': "Split-\nbag\nShuffle",
    'split-bag-k-fold': "Split-\nbag\nK-fold",
    'full-bag-stratified-k-fold': "Full-\nbag\nK-fold"
}

error_metric_map = {
    "error_bag_abs": "Abs loss",
    "hypergeo": "Prob. based loss"
}

final_results = pd.read_parquet("kdd-experiment-results.parquet")
final_results.rename(columns={"metric": "error_metric"}, inplace=True)

final_results["error_metric"].replace(error_legend_map, inplace=True)
final_results["split_method"].replace(split_method_map, inplace=True)

final_results["split_method"] = final_results["split_method"] + " " + final_results["validation_size_perc"].astype(str)
final_results["split_method"] = final_results["split_method"].str.replace("nan", "")

final_results["dataset"] = final_results["dataset"] + "-" + final_results["n_splits"] + "folds"

# Removing some error metrics (using only abs in the paper)
final_results = final_results[final_results["error_metric"].isin(["abs"])]
final_results["error_metric"].replace(error_metric_map, inplace=True)

base_datasets = ["CIFAR-10-Grey-Airplane-Automobile", "CIFAR-10-Grey-Airplane-Frog",
                    "MNIST-Digits-4-9", "MNIST-Digits-5-8", "Adult", "Default-Credit-Card", 
                    "Covertype", "Census-KDD", "MNIST-Digits-2-8", "MNIST-Digits-7-9",
                    "CIFAR-10-Grey-Frog-Truck", "CIFAR-10-Grey-Deer-Ship",
                    "MNIST-Digits-6-7", "MNIST-Digits-1-4", "MNIST-Digits-0-4", "MNIST-Digits-0-1",
                    "MNIST-Digits-3-5", "CIFAR-10-Grey-Frog-Ship", "CIFAR-10-Grey-Bird-Ship", 
                    "CIFAR-10-Grey-Bird-Cat", "CIFAR-10-Grey-Airplane-Horse", "CIFAR-10-Grey-Cat-Truck",
                    "CIFAR-10-Grey-Dog-Frog", "CIFAR-10-Grey-Cat-Horse", "CIFAR-10-Grey-Automobile-Dog", 
                    "MNIST-Digits-1-8", "MNIST-Digits-2-3", "MNIST-Digits-3-8"
                ]

base_datasets_type = {
    "CIFAR-10-Grey": "Image-Objects",
    "MNIST-Digits": "Image-Digits", 
    "Adult": "Tabular", 
    "Default-Credit-Card": "Tabular", 
    "Covertype": "Tabular", 
    "Census-KDD": "Tabular",
}

# # # TODO: fix
# base_datasets_gap = {
#     "CIFAR-10-Grey-Frog-Truck": "<= 0.1",
#     "CIFAR-10-Grey-Deer-Ship": "> 0.1 and < 0.2",
#     "CIFAR-10-Grey-Airplane-Automobile": ">= 0.2",
#     "CIFAR-10-Grey-Airplane-Frog": "<= 0.1",
#     "CIFAR-10-Grey-Frog-Ship": "<= 0.1",
#     "CIFAR-10-Grey-Bird-Ship": "> 0.1 and < 0.2",
#     "CIFAR-10-Grey-Bird-Cat": ">= 0.2",
#     "CIFAR-10-Grey-Airplane-Horse": ">= 0.2",
#     "CIFAR-10-Grey-Cat-Truck": "> 0.1 and < 0.2"
# }

dataset_map = {
    "cifar-10-grey-airplane-automobile-hard-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Airplane-Automobile (hard)\n",
    "cifar-10-grey-airplane-automobile-intermediate-3bags-cluster-kmeans-3-3folds" : "CIFAR-10-Grey-Airplane-Automobile\n (intermediate)",
    "cifar-10-grey-airplane-automobile-simple-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Airplane-Automobile (simple)\n",
    "cifar-10-grey-airplane-automobile-naive-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Airplane-Automobile (naive)\n",

    "cifar-10-grey-airplane-frog-hard-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Airplane-Frog (hard)\n",
    "cifar-10-grey-airplane-frog-intermediate-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Airplane-Frog (intermediate)\n",
    "cifar-10-grey-airplane-frog-simple-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Airplane-Frog (simple)\n",
    "cifar-10-grey-airplane-frog-naive-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Airplane-Frog (naive)\n",

    "mnist-digits-4-9-hard-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-4-9 (hard)\n",
    "mnist-digits-4-9-intermediate-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-4-9 (intermediate)\n",
    "mnist-digits-4-9-simple-3bags-cluster-None-None-3folds": "MNIST-Digits-4-9 (simple)\n",
    "mnist-digits-4-9-naive-3bags-cluster-None-None-3folds": "MNIST-Digits-4-9 (naive)\n",

    "mnist-digits-5-8-hard-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-5-8 (hard)\n",
    "mnist-digits-5-8-intermediate-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-5-8 (intermediate)\n",
    "mnist-digits-5-8-simple-3bags-cluster-None-None-3folds": "MNIST-Digits-5-8 (simple)\n",
    "mnist-digits-5-8-naive-3bags-cluster-None-None-3folds": "MNIST-Digits-5-8 (naive)\n",

    "adult-hard-10bags-cluster-kmeans-10-10folds": "Adult (hard)\n",
    "adult-intermediate-10bags-cluster-kmeans-10-10folds": "Adult (intermediate)\n",
    "adult-simple-10bags-cluster-None-None-10folds": "Adult (simple)\n",
    "adult-naive-10bags-cluster-None-None-10folds": "Adult (naive)\n",

    "default-credit-card-hard-5bags-cluster-kmeans-5-5folds": "Default-Credit-Card (hard)\n",
    "default-credit-card-intermediate-5bags-cluster-kmeans-5-5folds": "Default-Credit-Card (intermediate)\n",
    "default-credit-card-simple-5bags-cluster-None-None-5folds": "Default-Credit-Card (simple)\n",
    "default-credit-card-naive-5bags-cluster-None-None-5folds": "Default-Credit-Card (naive)\n",

    "covertype-classes-1-2-hard-10bags-cluster-kmeans-10-10folds": "Covertype (hard)\n",
    "covertype-classes-1-2-intermediate-10bags-cluster-kmeans-10-10folds": "Covertype (intermediate)\n",
    "covertype-classes-1-2-simple-10bags-cluster-None-None-10folds": "Covertype (simple)\n",
    "covertype-classes-1-2-naive-10bags-cluster-None-None-10folds": "Covertype (naive)\n",

    "census-kdd-balanced-hard-3bags-cluster-kmeans-3-3folds": "Census-KDD (hard)\n",
    "census-kdd-balanced-intermediate-3bags-cluster-kmeans-3-3folds": "Census-KDD (intermediate)\n",
    "census-kdd-balanced-simple-3bags-cluster-None-None-3folds": "Census-KDD (simple)\n",
    "census-kdd-balanced-naive-3bags-cluster-None-None-3folds": "Census-KDD (naive)\n",

    "mnist-digits-2-8-naive-3bags-cluster-None-None-3folds": "MNIST-Digits-2-8 (naive)\n",
    "mnist-digits-2-8-simple-3bags-cluster-None-None-3folds": "MNIST-Digits-2-8 (simple)\n",
    "mnist-digits-2-8-intermediate-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-2-8 (intermediate)\n",
    "mnist-digits-2-8-hard-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-2-8 (hard)\n",

    "mnist-digits-7-9-naive-3bags-cluster-None-None-3folds": "MNIST-Digits-7-9 (naive)\n",
    "mnist-digits-7-9-simple-3bags-cluster-None-None-3folds": "MNIST-Digits-7-9 (simple)\n",
    "mnist-digits-7-9-intermediate-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-7-9 (intermediate)\n",
    "mnist-digits-7-9-hard-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-7-9 (hard)\n",
    
    "cifar-10-grey-frog-truck-naive-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Frog-Truck (naive)\n",
    "cifar-10-grey-frog-truck-simple-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Frog-Truck (simple)\n",
    "cifar-10-grey-frog-truck-intermediate-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Frog-Truck (intermediate)\n",
    "cifar-10-grey-frog-truck-hard-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Frog-Truck (hard)\n",

    "cifar-10-grey-deer-ship-naive-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Deer-Ship (naive)\n",
    "cifar-10-grey-deer-ship-simple-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Deer-Ship (simple)\n",
    "cifar-10-grey-deer-ship-intermediate-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Deer-Ship (intermediate)\n",
    "cifar-10-grey-deer-ship-hard-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Deer-Ship (hard)\n",

    "mnist-digits-6-7-hard-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-6-7 (hard)\n",
    "mnist-digits-6-7-intermediate-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-6-7 (intermediate)\n",
    "mnist-digits-6-7-simple-3bags-cluster-None-None-3folds": "MNIST-Digits-6-7 (simple)\n",
    "mnist-digits-6-7-naive-3bags-cluster-None-None-3folds": "MNIST-Digits-6-7 (naive)\n",

    "mnist-digits-1-4-hard-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-1-4 (hard)\n",
    "mnist-digits-1-4-intermediate-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-1-4 (intermediate)\n",
    "mnist-digits-1-4-simple-3bags-cluster-None-None-3folds": "MNIST-Digits-1-4 (simple)\n",
    "mnist-digits-1-4-naive-3bags-cluster-None-None-3folds": "MNIST-Digits-1-4 (naive)\n",

    "mnist-digits-0-4-hard-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-0-4 (hard)\n",
    "mnist-digits-0-4-intermediate-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-0-4 (intermediate)\n",
    "mnist-digits-0-4-simple-3bags-cluster-None-None-3folds": "MNIST-Digits-0-4 (simple)\n",
    "mnist-digits-0-4-naive-3bags-cluster-None-None-3folds": "MNIST-Digits-0-4 (naive)\n",

    "mnist-digits-0-1-hard-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-0-1 (hard)\n",
    "mnist-digits-0-1-intermediate-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-0-1 (intermediate)\n",
    "mnist-digits-0-1-simple-3bags-cluster-None-None-3folds": "MNIST-Digits-0-1 (simple)\n",
    "mnist-digits-0-1-naive-3bags-cluster-None-None-3folds": "MNIST-Digits-0-1 (naive)\n",

    "cifar-10-grey-frog-ship-hard-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Frog-Ship (hard)\n",
    "cifar-10-grey-frog-ship-intermediate-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Frog-Ship (intermediate)\n",
    "cifar-10-grey-frog-ship-simple-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Frog-Ship (simple)\n",
    "cifar-10-grey-frog-ship-naive-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Frog-Ship (naive)\n",

    "cifar-10-grey-bird-ship-hard-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Bird-Ship (hard)\n",
    "cifar-10-grey-bird-ship-intermediate-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Bird-Ship (intermediate)\n",
    "cifar-10-grey-bird-ship-simple-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Bird-Ship (simple)\n",
    "cifar-10-grey-bird-ship-naive-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Bird-Ship (naive)\n",

    "cifar-10-grey-bird-cat-hard-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Bird-Cat (hard)\n",
    "cifar-10-grey-bird-cat-intermediate-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Bird-Cat (intermediate)\n",
    "cifar-10-grey-bird-cat-simple-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Bird-Cat (simple)\n",
    "cifar-10-grey-bird-cat-naive-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Bird-Cat (naive)\n",

    "cifar-10-grey-airplane-horse-hard-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Airplane-Horse (hard)\n",
    "cifar-10-grey-airplane-horse-intermediate-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Airplane-Horse (intermediate)\n",
    "cifar-10-grey-airplane-horse-simple-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Airplane-Horse (simple)\n",
    "cifar-10-grey-airplane-horse-naive-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Airplane-Horse (naive)\n",

    "mnist-digits-3-5-hard-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-3-5 (hard)\n",
    "mnist-digits-3-5-intermediate-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-3-5 (intermediate)\n",
    "mnist-digits-3-5-simple-3bags-cluster-None-None-3folds": "MNIST-Digits-3-5 (simple)\n",
    "mnist-digits-3-5-naive-3bags-cluster-None-None-3folds": "MNIST-Digits-3-5 (naive)\n",

    "cifar-10-grey-cat-truck-hard-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Cat-Truck (hard)\n",
    "cifar-10-grey-cat-truck-intermediate-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Cat-Truck (intermediate)\n",
    "cifar-10-grey-cat-truck-simple-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Cat-Truck (simple)\n",
    "cifar-10-grey-cat-truck-naive-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Cat-Truck (naive)\n",

    "cifar-10-grey-dog-frog-hard-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Dog-Frog (hard)\n",
    "cifar-10-grey-dog-frog-intermediate-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Dog-Frog (intermediate)\n",
    "cifar-10-grey-dog-frog-simple-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Dog-Frog (simple)\n",
    "cifar-10-grey-dog-frog-naive-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Dog-Frog (naive)\n",

    "cifar-10-grey-cat-horse-hard-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Cat-Horse (hard)\n",
    "cifar-10-grey-cat-horse-intermediate-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Cat-Horse (intermediate)\n",
    "cifar-10-grey-cat-horse-simple-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Cat-Horse (simple)\n",
    "cifar-10-grey-cat-horse-naive-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Cat-Horse (naive)\n",

    "cifar-10-grey-automobile-dog-hard-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Automobile-Dog (hard)\n",
    "cifar-10-grey-automobile-dog-intermediate-3bags-cluster-kmeans-3-3folds": "CIFAR-10-Grey-Automobile-Dog (intermediate)\n",
    "cifar-10-grey-automobile-dog-simple-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Automobile-Dog (simple)\n",
    "cifar-10-grey-automobile-dog-naive-3bags-cluster-None-None-3folds": "CIFAR-10-Grey-Automobile-Dog (naive)\n",

    "mnist-digits-1-8-hard-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-1-8 (hard)\n",
    "mnist-digits-1-8-intermediate-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-1-8 (intermediate)\n",
    "mnist-digits-1-8-simple-3bags-cluster-None-None-3folds": "MNIST-Digits-1-8 (simple)\n",
    "mnist-digits-1-8-naive-3bags-cluster-None-None-3folds": "MNIST-Digits-1-8 (naive)\n",

    "mnist-digits-2-3-hard-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-2-3 (hard)\n",
    "mnist-digits-2-3-intermediate-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-2-3 (intermediate)\n",
    "mnist-digits-2-3-simple-3bags-cluster-None-None-3folds": "MNIST-Digits-2-3 (simple)\n",
    "mnist-digits-2-3-naive-3bags-cluster-None-None-3folds": "MNIST-Digits-2-3 (naive)\n",

    "mnist-digits-3-8-hard-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-3-8 (hard)\n",
    "mnist-digits-3-8-intermediate-3bags-cluster-kmeans-3-3folds": "MNIST-Digits-3-8 (intermediate)\n",
    "mnist-digits-3-8-simple-3bags-cluster-None-None-3folds": "MNIST-Digits-3-8 (simple)\n",
    "mnist-digits-3-8-naive-3bags-cluster-None-None-3folds": "MNIST-Digits-3-8 (naive)\n",
}

model_map = {
    "lmm": "LMM",
    "llp-svm-lin": "Alter-SVM",
    "kdd-lr": "EM/LR",
}

final_results["model"].replace(model_map, inplace=True)
final_results["dataset"].replace(dataset_map, inplace=True)

# Creating a column with the dataset variant
final_results["dataset_variant"] = final_results["dataset"].apply(get_dataset_variant)

# Creating a columns with the base dataset
final_results["base_dataset"] = "None"
for dataset in base_datasets:
    final_results.loc[final_results.dataset.str.contains(dataset), "base_dataset"] = dataset

final_results["dataset_type"] = "None"
for base_dataset in base_datasets_type:
    final_results.loc[final_results.base_dataset.str.contains(base_dataset), "dataset_type"] = base_datasets_type[base_dataset]

if args.plot_type == "winning-figures" or args.plot_type == "table-results" or args.plot_type == "effect-sizes":
    # Computing the W/L/D
    winning_df_shuffle = pd.DataFrame(columns=["dataset", "algorithm", "win", "lose", "draw"])
    winning_df_bootstrap = pd.DataFrame(columns=["dataset", "algorithm", "win", "lose", "draw"])
    winning_df_kfold = pd.DataFrame(columns=["dataset", "algorithm", "win", "lose", "draw"])

    for dataset in final_results.dataset.unique():
        for model in final_results.model.unique():
            accuracy_test_split_method = {}
            for split_method in final_results.split_method.unique():
                X = final_results[(final_results.dataset == dataset) & (final_results.model == model) & (final_results.split_method == split_method)]
                # Just add if X is not empty
                if len(X) > 0:
                    accuracy_test_split_method[split_method] = X.accuracy_test.values
            
            # Verify if accuracy_test_split_method is empty
            if len(accuracy_test_split_method) == 0:
                continue
            
            # If Full-bag K-fold is not in the accuracy_test_split_method, continue
            if "Full-\nbag\nK-fold " not in accuracy_test_split_method:
                continue
            
            # Test if the results are statistically significant

            # Bootstrap
            if "Split-\nBag\nBootstrap 0.5" in accuracy_test_split_method:
                sp_bootstrap_test = ttest_ind(accuracy_test_split_method["Full-\nbag\nK-fold "],
                                                accuracy_test_split_method["Split-\nBag\nBootstrap 0.5"], 
                                                equal_var=False, random_state=73921)
                if sp_bootstrap_test.pvalue <= 0.05:
                    # Win the one with the highest average accuracy
                    if np.mean(accuracy_test_split_method["Full-\nbag\nK-fold "]) > np.mean(accuracy_test_split_method["Split-\nBag\nBootstrap 0.5"]):
                        # Full bag wins
                        winning_df_bootstrap = pd.concat([pd.DataFrame([[dataset, model, 0, 1, 0]], columns=winning_df_bootstrap.columns), 
                                                            winning_df_bootstrap], ignore_index=True)
                    else:
                        # Split bag wins
                        winning_df_bootstrap = pd.concat([pd.DataFrame([[dataset, model, 1, 0, 0]], columns=winning_df_bootstrap.columns), 
                                                            winning_df_bootstrap], ignore_index=True)
                else:
                    # Draw
                    winning_df_bootstrap = pd.concat([pd.DataFrame([[dataset, model, 0, 0, 1]], columns=winning_df_bootstrap.columns), 
                                                        winning_df_bootstrap], ignore_index=True)

            # Shuffle
            if "Split-\nbag\nShuffle 0.5" in accuracy_test_split_method:
                sp_shuffle_test = ttest_ind(accuracy_test_split_method["Full-\nbag\nK-fold "],
                                                accuracy_test_split_method["Split-\nbag\nShuffle 0.5"],
                                                equal_var=False, random_state=73921)
                if sp_shuffle_test.pvalue <= 0.05:
                    # Win the one with the highest average accuracy
                    if np.mean(accuracy_test_split_method["Full-\nbag\nK-fold "]) > np.mean(accuracy_test_split_method["Split-\nbag\nShuffle 0.5"]):
                        # Full bag wins
                        winning_df_shuffle = pd.concat([pd.DataFrame([[dataset, model, 0, 1, 0]], columns=winning_df_shuffle.columns), 
                                                            winning_df_shuffle], ignore_index=True)
                    else:
                        # Split bag wins
                        winning_df_shuffle = pd.concat([pd.DataFrame([[dataset, model, 1, 0, 0]], columns=winning_df_shuffle.columns), 
                                                            winning_df_shuffle], ignore_index=True)
                else:
                    # Draw
                    winning_df_shuffle = pd.concat([pd.DataFrame([[dataset, model, 0, 0, 1]], columns=winning_df_shuffle.columns), 
                                                        winning_df_shuffle], ignore_index=True)

            # K-fold
            if "Split-\nbag\nK-fold " in accuracy_test_split_method:
                sp_kfold_test = ttest_ind(accuracy_test_split_method["Full-\nbag\nK-fold "],
                                                accuracy_test_split_method["Split-\nbag\nK-fold "],
                                                equal_var=False, random_state=73921)
                if sp_kfold_test.pvalue <= 0.05:
                    # Win the one with the highest average accuracy
                    if np.mean(accuracy_test_split_method["Full-\nbag\nK-fold "]) > np.mean(accuracy_test_split_method["Split-\nbag\nK-fold "]):
                        # Full bag wins
                        winning_df_kfold = pd.concat([pd.DataFrame([[dataset, model, 0, 1, 0]], columns=winning_df_kfold.columns), 
                                                            winning_df_kfold], ignore_index=True)
                    else:
                        # Split bag wins
                        winning_df_kfold = pd.concat([pd.DataFrame([[dataset, model, 1, 0, 0]], columns=winning_df_kfold.columns), 
                                                            winning_df_kfold], ignore_index=True)
                else:
                    # Draw
                    winning_df_kfold = pd.concat([pd.DataFrame([[dataset, model, 0, 0, 1]], columns=winning_df_kfold.columns), 
                                                        winning_df_kfold], ignore_index=True)
    # Creating a column with the dataset variant
    winning_df_bootstrap["dataset_variant"] = winning_df_bootstrap["dataset"].apply(get_dataset_variant)
    winning_df_shuffle["dataset_variant"] = winning_df_shuffle["dataset"].apply(get_dataset_variant)
    winning_df_kfold["dataset_variant"] = winning_df_kfold["dataset"].apply(get_dataset_variant)

    # Creating a columns with the base dataset
    winning_df_bootstrap["base_dataset"] = "None"
    winning_df_shuffle["base_dataset"] = "None"
    winning_df_kfold["base_dataset"] = "None"
    for dataset in base_datasets:
        winning_df_bootstrap.loc[winning_df_bootstrap.dataset.str.contains(dataset), "base_dataset"] = dataset
        winning_df_shuffle.loc[winning_df_shuffle.dataset.str.contains(dataset), "base_dataset"] = dataset
        winning_df_kfold.loc[winning_df_kfold.dataset.str.contains(dataset), "base_dataset"] = dataset

    # Creating a column with the dataset type using base_datasets_type
    winning_df_bootstrap["dataset_type"] = "None"
    winning_df_shuffle["dataset_type"] = "None"
    winning_df_kfold["dataset_type"] = "None"
    for base_dataset in base_datasets_type:
        winning_df_bootstrap.loc[winning_df_bootstrap.base_dataset.str.contains(base_dataset), "dataset_type"] = base_datasets_type[base_dataset]
        winning_df_shuffle.loc[winning_df_shuffle.base_dataset.str.contains(base_dataset), "dataset_type"] = base_datasets_type[base_dataset]
        winning_df_kfold.loc[winning_df_kfold.base_dataset.str.contains(base_dataset), "dataset_type"] = base_datasets_type[base_dataset]
    
    winning_df_bootstrap["split_method"] = split_method_map["split-bag-bootstrap"]
    winning_df_shuffle["split_method"] = split_method_map["split-bag-shuffle"]
    winning_df_kfold["split_method"] = split_method_map["split-bag-k-fold"]

if args.plot_type == "check-n-experiments":
    total_experiments = len(final_results)
    print("Total experiments: ", total_experiments)
    for model in final_results["model"].unique():
        print(model, len(final_results[final_results["model"] == model]))

    x = final_results[final_results["model"] == "Alter-SVM"]
    for dataset in x.dataset.unique():
        y = x[x["dataset"] == dataset]
        if y.split_method.nunique() != 3:
            print(dataset, y.split_method.unique())

    for base_dataset in x["base_dataset"].unique():
        y = x[x["base_dataset"] == base_dataset]
        if y.dataset_variant.nunique() != 3:
            print(base_dataset, y.dataset_variant.unique())

elif args.plot_type == "winning-figures":
    winning_df = pd.concat([winning_df_bootstrap, winning_df_shuffle, winning_df_kfold], ignore_index=True)
    N_per_variant = winning_df.dataset_variant.value_counts()
    N_per_dataset_type = winning_df.dataset_type.value_counts()
    N_per_algorithm = winning_df.algorithm.value_counts()

    # Print the overall fraction of winning in the significant tests
    significant_tests = winning_df[(winning_df.win == 1) | (winning_df.lose == 1)]
    frac_win = significant_tests[significant_tests.win == 1].shape[0] / significant_tests.shape[0]
    print("Overall fraction of winning in the significant tests: %.2f" % (frac_win))

    # Main figure of the paper (W/L/D percentage per dataset variant)
    winning_df_variant_sm = winning_df.groupby(["dataset_variant", "split_method"]).sum()[["win", "lose", "draw"]]
    # Rename the index from variant to variant + N (number of datasets)
    for variant in winning_df_variant_sm.index.get_level_values(0):    
        winning_df_variant_sm.rename(index={variant: variant + "\n(N=" + str(N_per_variant[variant]) + ")"}, inplace=True)
    winning_df_variant_sm = winning_df_variant_sm.div(winning_df_variant_sm.sum(axis=1), axis=0)
    winning_df_variant_sm.rename(columns={"win": "S > F", "lose": "S < F", "draw": "No significance"}, inplace=True)

    filename = "plots/aggregate_result_per_variant.pdf"
    plot_winning_figure(winning_df_variant_sm, filename, plot_type="variants")
    
    # # Per dataset type
    winning_df_type_sm = winning_df.groupby(["dataset_type", "split_method"]).sum()[["win", "lose", "draw"]]
    for dataset_type in winning_df_type_sm.index.get_level_values(0):    
        winning_df_type_sm.rename(index={dataset_type: dataset_type + "\n(N=" + str(N_per_dataset_type[dataset_type]) + ")"}, inplace=True)
    winning_df_type_sm = winning_df_type_sm.div(winning_df_type_sm.sum(axis=1), axis=0)
    winning_df_type_sm.rename(columns={"win": "S > F", "lose": "S < F", "draw": "No significance"}, inplace=True)

    filename = "plots/aggregate_result_per_dataset_type.pdf"
    plot_winning_figure(winning_df_type_sm, filename)

    # Per algorithm
    winning_df_algorithm_sm = winning_df.groupby(["algorithm", "split_method"]).sum()[["win", "lose", "draw"]]
    for algorithm in winning_df_algorithm_sm.index.get_level_values(0):
        winning_df_algorithm_sm.rename(index={algorithm: algorithm + "\n(N=" + str(N_per_algorithm[algorithm]) + ")"}, inplace=True)
    winning_df_algorithm_sm = winning_df_algorithm_sm.div(winning_df_algorithm_sm.sum(axis=1), axis=0)
    winning_df_algorithm_sm.rename(columns={"win": "S > F", "lose": "S < F", "draw": "No significance"}, inplace=True)

    filename = "plots/aggregate_result_per_algorithm.pdf"
    plot_winning_figure(winning_df_algorithm_sm, filename, plot_type="algorithms")

elif args.plot_type == "hyperparams-plots":
    # Hyperparams plot (alter-SVM)
    final_results_alter_svm = deepcopy(final_results[final_results.model == "Alter-SVM"])
    final_results_alter_svm["C"] = final_results_alter_svm.best_hyperparams.apply(lambda x: x["C"])
    final_results_alter_svm["C_p"] = final_results_alter_svm.best_hyperparams.apply(lambda x: x["C_p"])

    # Plotting this for one dataset only
    dataset = final_results_alter_svm.dataset.unique()[0]
    fontsize = 5
    fig, ax = plt.subplots(3, 4, sharex=False, sharey=False, gridspec_kw={"height_ratios": [6, 1, 4]}, figsize=(3.5, 3.5))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    cbar_ax.tick_params(labelsize=fontsize)

    split_methods = sorted(final_results.split_method.unique())
    split_methods[2], split_methods[3] = split_methods[3], split_methods[2]

    for i, sm in enumerate(split_methods):
        X = deepcopy(final_results_alter_svm[(final_results_alter_svm.dataset == dataset) & (final_results_alter_svm.split_method == sm)])
        if X.shape[0] == 0:
            continue
        X.drop(columns=["exec"], inplace=True)
        X["C"] = X["C"].astype(str)
        X["C_p"] = X["C_p"].astype(str)
        X = pd.crosstab(X["C"], X["C_p"])

        df = pd.DataFrame(data = np.zeros((6, 6), int),
                            columns = [str(x) for x in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]], 
                            index = [str(x) for x in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]])
        df = df.rename_axis(index="C", columns="C_p")
        df = (df + X).fillna(0)
        if i == 2:
            sns.heatmap(df, ax=ax[0, i], vmin=0, vmax=10, cbar=True, cbar_ax=cbar_ax, cmap="icefire", xticklabels=True, yticklabels=True)
        else:
            sns.heatmap(df, ax=ax[0, i], vmin=0, vmax=10, cbar=False, cmap="icefire", xticklabels=True, yticklabels=True)
        ax[0, i].set_title(sm.replace("0.5", ""), size=fontsize+2)
    
    tl = [r"$10^{-2}$", r"$10^{-1}$", r"$10^0$", r"$10^{1}$", r"$10^{2}$", r"$10^{3}$"]
    tl_x = [r"$10^{-2}$", "", "", "",  "", r"$10^{3}$"]

    ax[0, 3].axis('off')
    ax[0, 3].set_title(split_methods[3].replace("0.5", ""), size=fontsize+2)

    for i in range(3):
        ax[0, i].set_xlabel(r"$C_p$", size=fontsize, labelpad=-4)
        ax[0, i].set_xticklabels(tl_x, rotation=0, size=fontsize)
        ax[0, i].set_ylabel("")
        ax[0, i].set_yticklabels([])

    ax[0, 0].set_yticklabels(tl, size=fontsize)
    ax[0, 0].set_ylabel(r"$C$", size=fontsize)

    # EM/LR
    final_results_kdd_lr = deepcopy(final_results[final_results.model == "EM/LR"])
    final_results_kdd_lr["C"] = final_results_kdd_lr.best_hyperparams.apply(lambda x: x["C"])

    for i, sm in enumerate(split_methods):
        X = deepcopy(final_results_kdd_lr[(final_results_kdd_lr.dataset == dataset) & (final_results_kdd_lr.split_method == sm)])
        X.drop(columns=["exec"], inplace=True)
        X["C"] = X["C"].astype(str)
        
        df = pd.DataFrame(data = np.zeros((1, 6), int),
                            columns = [str(x) for x in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]])
        df = df.rename_axis(columns="C")
        for C_value, count in X.C.value_counts().items():
            df[C_value] = count

        if i == 2:
            sns.heatmap(df, ax=ax[1, i], vmin=0, vmax=30, cbar=True, cbar_ax=cbar_ax, cmap="icefire", xticklabels=True, yticklabels=True)
        else:
            sns.heatmap(df, ax=ax[1, i], vmin=0, vmax=30, cbar=False, cmap="icefire", xticklabels=True, yticklabels=True)

    for i in range(4):
        ax[1, i].set_xlabel(r"$C$", size=fontsize, labelpad=-4)
        ax[1, i].set_xticklabels(tl_x, rotation=0, size=fontsize)
        ax[1, i].set_ylabel("")
        ax[1, i].set_yticklabels([])

    # LMM
    final_results_lmm = deepcopy(final_results[final_results.model == "LMM"])
    # Getting only two/three hyperparameters
    final_results_lmm["lambda"] = final_results_lmm.best_hyperparams.apply(lambda x: x["lambda"])
    final_results_lmm["sigma"] = final_results_lmm.best_hyperparams.apply(lambda x: x["sigma"])

    for i, sm in enumerate(split_methods):
        X = deepcopy(final_results_lmm[(final_results_lmm.dataset == dataset) & (final_results_lmm.split_method == sm)])
        X.drop(columns=["exec"], inplace=True)
        X["lambda"] = X["lambda"].astype(str)
        X["sigma"] = X["sigma"].astype(str)
        X = pd.crosstab(X["lambda"], X["sigma"])

        df = pd.DataFrame(data = np.zeros((4, 3), int),
                            index = [str(x) for x in [0.0, 1.0, 10.0, 100.0]], 
                            columns = [str(x) for x in [0.25, 0.5, 1.0]])
        df = df.rename_axis(index="lambda", columns="sigma")
        df = (df + X).fillna(0)
        if i == 2:
            sns.heatmap(df, ax=ax[2, i], vmin=0, vmax=30, cbar=True, cbar_ax=cbar_ax, cmap="icefire", xticklabels=True, yticklabels=True)
        else:
            sns.heatmap(df, ax=ax[2, i], vmin=0, vmax=30, cbar=False, cmap="icefire", xticklabels=True, yticklabels=True)

    #plt.suptitle(dataset, size=10)
    tl = [r"$2^{-2}$", r"$2^{-1}$", r"$2^{0}$"]

    for i in range(4):
        ax[2, i].set_xlabel(r"$\sigma$", size=fontsize)
        ax[2, i].set_xticklabels(tl, rotation=0, size=fontsize)
        ax[2, i].set_ylabel("")
        ax[2, i].set_yticklabels([])

    ax[2, 0].set_yticklabels([r"$0$", r"$10^0$", r"$10^1$", r"$10^2$"], size=fontsize)
    ax[2, 0].set_ylabel(r"$\lambda$", size=fontsize)

    fig.supylabel("LMM | EM/LR | SVM", size=fontsize+2, rotation=90)

    for i in range(4):
        ax[0, i].tick_params(axis='both', which='major', length=1, width=1)
        ax[1, i].tick_params(axis='both', which='major', length=1, width=1)
        ax[2, i].tick_params(axis='both', which='major', length=1, width=1)

    plt.savefig("plots/{}_hyperparams.pdf".format(dataset.replace("\n", "")), bbox_inches='tight', pad_inches=0.01, dpi=800)

elif args.plot_type == "effect-sizes":
    winning_df = pd.concat([winning_df_bootstrap, winning_df_shuffle, winning_df_kfold], ignore_index=True)

    # Start effect size plot
    significant_tests_win = winning_df[(winning_df.win == 1)]

    diffs = []
    for row in significant_tests_win.itertuples():
        dataset = row.dataset
        model = row.algorithm
        split_method = row.split_method
        X_row = final_results[(final_results.dataset == dataset) & (final_results.model == model)]
        X_row_split_method = X_row[X_row.split_method.str.contains(split_method)]
        X_row_full_bag = X_row[X_row.split_method.str.contains("Full")]
        if not (len(X_row_split_method) in [10, 30]) or not (len(X_row_full_bag) in [10, 30]):
            print("Dataset: %s, model: %s, split_method: %s" % (dataset, model, split_method))
            print("X_row_split_method: %s" % (X_row_split_method))
            print("X_row_full_bag: %s" % (X_row_full_bag))
            continue
        
        diffs.append(X_row_split_method.accuracy_test.mean() - X_row_full_bag.accuracy_test.mean())


    significant_tests_lose = winning_df[(winning_df.lose == 1)]

    for row in significant_tests_lose.itertuples():
        dataset = row.dataset
        model = row.algorithm
        split_method = row.split_method
        X_row = final_results[(final_results.dataset == dataset) & (final_results.model == model)]
        X_row_split_method = X_row[X_row.split_method.str.contains(split_method)]
        X_row_full_bag = X_row[X_row.split_method.str.contains("Full")]
        if not (len(X_row_split_method) in [10, 30]) or not (len(X_row_full_bag) in [10, 30]):
            print("Dataset: %s, model: %s, split_method: %s" % (dataset, model, split_method))
            print("X_row_split_method: %s" % (X_row_split_method))
            print("X_row_full_bag: %s" % (X_row_full_bag))
            continue
        
        diffs.append(X_row_split_method.accuracy_test.mean() - X_row_full_bag.accuracy_test.mean())

    # Effect sizes plot
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('font', size=6)
    
    _, ax = plt.subplots(figsize=(3.5, 2))

    g = sns.histplot(diffs, kde=True, ax=ax)
    plt.xlabel("Difference in accuracy")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("plots/effect-sizes-dist.pdf", bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

elif args.plot_type == "table-results":
    final_results_shuffle = deepcopy(final_results[(final_results.split_method == split_method_map["split-bag-shuffle"] + " 0.5") | \
                                        (final_results.split_method == split_method_map["full-bag-stratified-k-fold"] + " ")])
    final_results_bootstrap = deepcopy(final_results[(final_results.split_method == split_method_map["split-bag-bootstrap"] + " 0.5") | \
                                        (final_results.split_method == split_method_map["full-bag-stratified-k-fold"] + " ")])
    final_results_kfold = deepcopy(final_results[(final_results.split_method == split_method_map["split-bag-k-fold"] + " ") | \
                                        (final_results.split_method == split_method_map["full-bag-stratified-k-fold"] + " ")])

    winning_df_bootstrap.split_method.replace({split_method_map["split-bag-bootstrap"]: "Split-bag Bootstrap"}, inplace=True)
    winning_df_shuffle.split_method.replace({split_method_map["split-bag-shuffle"]: "Split-bag Shuffle"}, inplace=True)
    winning_df_kfold.split_method.replace({split_method_map["split-bag-k-fold"]: "Split-bag K-fold"}, inplace=True)

    # Shuffle table
    final_results_shuffle.split_method.replace({split_method_map["split-bag-shuffle"] + " 0.5": "Split-bag Shuffle", split_method_map["full-bag-stratified-k-fold"] + " ": "Full-bag K-fold"}, inplace=True)
    table_shuffle = pd.pivot_table(final_results_shuffle, values='accuracy_test', index=['base_dataset', 'dataset_variant'],
                        columns=['model', "split_method"], aggfunc=lambda x: "%.4f (%.4f)" % (np.mean(x), 1.96 * np.std(x)/np.sqrt(len(x))), fill_value=np.nan)

    for i, row in winning_df_shuffle.iterrows():
        split_method = row.split_method
        variant = row.dataset_variant
        base_dataset = row.base_dataset
        algorithm = row.algorithm

        if row.win == 1:
            value = table_shuffle.loc[(base_dataset, variant), (algorithm, split_method)]
            table_shuffle.loc[(base_dataset, variant), (algorithm, split_method)] = "\\textbf{" + value + "}"
        elif row.lose == 1:
            value = table_shuffle.loc[(base_dataset, variant), (algorithm, "Full-bag K-fold")]
            table_shuffle.loc[(base_dataset, variant), (algorithm, "Full-bag K-fold")] = "\\textbf{" + value + "}"

    # Rename table index
    table_shuffle.index = table_shuffle.index.rename(["Base Dataset", "Variant"])
    # Rename table columns
    table_shuffle.columns = table_shuffle.columns.rename(["LLP Algorithm", "Split Algorithm"])

    with pd.option_context("max_colwidth", 10000):
        table_shuffle.to_latex(buf="tables/table-results-shuffle.tex", index=True, escape=False, na_rep="", sparsify=True)

    # Bootstrap table
    final_results_bootstrap.split_method.replace({split_method_map["split-bag-bootstrap"] + " 0.5": "Split-bag Bootstrap", split_method_map["full-bag-stratified-k-fold"] + " ": "Full-bag K-fold"}, inplace=True)
    table_bootstrap = pd.pivot_table(final_results_bootstrap, values='accuracy_test', index=['base_dataset', 'dataset_variant'],
                        columns=['model', "split_method"], aggfunc=lambda x: "%.4f (%.4f)" % (np.mean(x), 1.96 * np.std(x)/np.sqrt(len(x))), fill_value=np.nan)
    
    for i, row in winning_df_bootstrap.iterrows():
        split_method = row.split_method
        variant = row.dataset_variant
        base_dataset = row.base_dataset
        algorithm = row.algorithm

        if row.win == 1:
            value = table_bootstrap.loc[(base_dataset, variant), (algorithm, split_method)]
            table_bootstrap.loc[(base_dataset, variant), (algorithm, split_method)] = "\\textbf{" + value + "}"
        elif row.lose == 1:
            value = table_bootstrap.loc[(base_dataset, variant), (algorithm, "Full-bag K-fold")]
            table_bootstrap.loc[(base_dataset, variant), (algorithm, "Full-bag K-fold")] = "\\textbf{" + value + "}"
    
    # Rename table index
    table_bootstrap.index = table_bootstrap.index.rename(["Base Dataset", "Variant"])
    # Rename table columns
    table_bootstrap.columns = table_bootstrap.columns.rename(["LLP Algorithm", "Split Algorithm"])

    with pd.option_context("max_colwidth", 10000):
        table_bootstrap.to_latex(buf="tables/table-results-bootstrap.tex", index=True, escape=False, na_rep="", sparsify=True)
    
    # K-fold table
    final_results_kfold.split_method.replace({split_method_map["split-bag-k-fold"] + " ": "Split-bag K-fold", split_method_map["full-bag-stratified-k-fold"] + " ": "Full-bag K-fold"}, inplace=True)
    table_kfold = pd.pivot_table(final_results_kfold, values='accuracy_test', index=['base_dataset', 'dataset_variant'],
                        columns=['model', "split_method"], aggfunc=lambda x: "%.4f (%.4f)" % (np.mean(x), 1.96 * np.std(x)/np.sqrt(len(x))), fill_value=np.nan)
    
    for i, row in winning_df_kfold.iterrows():
        split_method = row.split_method
        variant = row.dataset_variant
        base_dataset = row.base_dataset
        algorithm = row.algorithm

        if row.win == 1:
            value = table_kfold.loc[(base_dataset, variant), (algorithm, split_method)]
            table_kfold.loc[(base_dataset, variant), (algorithm, split_method)] = "\\textbf{" + value + "}"
        elif row.lose == 1:
            value = table_kfold.loc[(base_dataset, variant), (algorithm, "Full-bag K-fold")]
            table_kfold.loc[(base_dataset, variant), (algorithm, "Full-bag K-fold")] = "\\textbf{" + value + "}"
        
    # Rename table index
    table_kfold.index = table_kfold.index.rename(["Base Dataset", "Variant"])
    # Rename table columns
    table_kfold.columns = table_kfold.columns.rename(["LLP Algorithm", "Split Algorithm"])

    with pd.option_context("max_colwidth", 10000):
        table_kfold.to_latex(buf="tables/table-results-kfold.tex", index=True, escape=False, na_rep="", sparsify=True)
elif args.plot_type == "datasets-info":
    dataset_info = pd.DataFrame(columns=["Dataset", "Number of bags", "Proportions", "Bag sizes"])
    for dataset in dataset_map:
        filename = "datasets-ci/" + "-".join(dataset.split("-")[:-1]) + ".parquet"
        df = pd.read_parquet(filename)
        proportions = compute_proportions(df.bag.values, df.y.values)
        proportions = [round(x, 2) for x in proportions]
        bags_sizes = [len(df[df.bag == bag]) for bag in sorted(df.bag.unique())]
        list2str = lambda x: ("(" + ",".join([str(y) for y in x]) + ")").replace(",)", ")")
        dataset_info = pd.concat([dataset_info, pd.DataFrame([[dataset_map[dataset].replace("\n", ""), len(df.bag.unique()), list2str(proportions), list2str(bags_sizes)]], columns=dataset_info.columns)], ignore_index=True)
    dataset_info.sort_values(by=["Dataset"], inplace=True)
    with pd.option_context("max_colwidth", 10000):
        dataset_info.to_latex(buf="tables/table-datasets-info.tex", index=False, escape=False)
elif args.plot_type == "base-datasets-info":
    df = pd.DataFrame(columns=["Dataset", "$N$", "Features", "\\begin{tabular}[c]{@{}c@{}}Proportions\\ of 1's\end{tabular}", "Source"])

    # Adult
    X = pd.read_parquet("datasets-ci/adult-hard-10bags-cluster-kmeans-10.parquet")
    df = pd.concat([df, pd.DataFrame([["Adult", len(X), len(X.columns) - 2, round(X.y.sum() / len(X.y), 2), "UCI"]], columns=df.columns)], ignore_index=True)

    # Default-Credit-Card
    X = pd.read_parquet("datasets-ci/default-credit-card-hard-5bags-cluster-kmeans-5.parquet")
    df = pd.concat([df, pd.DataFrame([["Default-Credit-Card", len(X), len(X.columns) - 2, round(X.y.sum() / len(X.y), 2), "UCI"]], columns=df.columns)], ignore_index=True)

    # Covertype
    X = pd.read_parquet("datasets-ci/covertype-classes-1-2-hard-10bags-cluster-kmeans-10.parquet")
    df = pd.concat([df, pd.DataFrame([["Covertype (Classes 1 and 2)", len(X), len(X.columns) - 2, round(X.y.sum() / len(X.y), 2), "UCI"]], columns=df.columns)], ignore_index=True)

    # Census-KDD
    X = pd.read_parquet("datasets-ci/census-kdd-balanced-hard-3bags-cluster-kmeans-3.parquet")
    df = pd.concat([df, pd.DataFrame([["Census-KDD (Balanced)", len(X), len(X.columns) - 2, round(X.y.sum() / len(X.y), 2), "UCI"]], columns=df.columns)], ignore_index=True)

    # MNIST-Digits
    X = pd.read_parquet("datasets-ci/mnist-digits-0-1-hard-3bags-cluster-kmeans-3.parquet")
    df = pd.concat([df, pd.DataFrame([["MNIST-Digits (Pairs of classes)", "Ranges from 13138 to 14780", len(X.columns) - 2, "Ranges from 0.49 to 0.54", "\cite{lecun2010mnist}"]], columns=df.columns)], ignore_index=True)

    # CIFAR-10-Grey
    X = pd.read_parquet("datasets-ci/cifar-10-grey-airplane-automobile-hard-3bags-cluster-kmeans-3.parquet")
    df = pd.concat([df, pd.DataFrame([["CIFAR-10 Grey (Pairs of classes)", len(X), len(X.columns) - 2, round(X.y.sum() / len(X.y), 2), "\cite{krizhevsky2009learning}"]], columns=df.columns)], ignore_index=True)

    with pd.option_context("max_colwidth", 10000):
        df.to_latex(buf="tables/table-base-datasets-info.tex", index=False, escape=False)
