from llp_learn.alter import alterSVM
from llp_learn.model_selection import gridSearchCV
from llp_learn.util import compute_proportions
from llp_learn.em import EM

from almostnolabel import LMM

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
from copy import deepcopy
import argparse
import os

from sklearn.model_selection import ShuffleSplit

def load_dataset(args):
    df = pd.read_parquet("datasets-ci/" + args.dataset + ".parquet")
    y = df["y"].values
    y = y.reshape(-1)
    y[y == 0] = -1
    bags = df["bag"].values
    bags = bags.reshape(-1)
    df.drop(columns=["y", "bag"], inplace=True)
    X = df.values
    train_index, test_index = next(ShuffleSplit(n_splits=1, test_size=0.25, random_state=seed[args.execution]).split(X))

    return X, bags, y, train_index, test_index

# Constants
n_executions = 30
try:
    N_JOBS = eval(os.getenv('NSLOTS'))
except:
    N_JOBS = -1
print("Using {} cores".format(N_JOBS))

seed = [189395, 962432364, 832061813, 316313123, 1090792484,
        1041300646,  242592193,  634253792,  391077503, 2644570296, 
        1925621443, 3585833024,  530107055, 3338766924, 3029300153,
       2924454568, 1443523392, 2612919611, 2781981831, 3394369024,
        641017724,  626917272, 1164021890, 3439309091, 1066061666,
        411932339, 1446558659, 1448895932,  952198910, 3882231031]

directory = "kdd-experiments/"

# Parsing arguments
parser = argparse.ArgumentParser(description="LLP loss experiments")
parser.add_argument("--dataset", "-d", required=True, help="the dataset that will be used in the experiments")
parser.add_argument("--model", "-m", choices=["llp-svm-lin", "kdd-lr", "lmm"], required=True,
                    help="the model that will be used in the experiments")
parser.add_argument("--loss", "-l", choices=["abs"],
                    help="the loss function that will be used in the experiment")
parser.add_argument("--n_splits", "-n", type=int,
                    help="the number of splits that will be used in the experiment")
parser.add_argument("--validation_size", "-v", type=float,
                    help="the validation size that will be used in the experiment")
parser.add_argument("--splitter", "-s", choices=["full-bag-stratified-k-fold", "split-bag-bootstrap", "split-bag-shuffle", "split-bag-k-fold"],
                    help="the splitter that will be used in the experiment")
parser.add_argument("--execution", "-e", choices=[x for x in range(n_executions)], type=int, required=True,
                    help="the execution of the experiment")
args = parser.parse_args()

try:
    os.mkdir(directory)
except:
    pass

if args.execution is not None:
    args.execution = int(args.execution)

filename = directory + str(args.dataset) + "_" + str(args.model) + "_" + str(
    args.loss) + "_" + str(None) + "_" + str(args.splitter) + "_" + str(args.n_splits) + "_" + str(args.validation_size) + "_" + str(args.execution) + ".parquet"

if args.model == "kdd-lr":
    params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}
elif args.model == "llp-svm-lin":
    params = {"C": [0.01, 0.1, 1, 10, 100, 1000], "C_p": [0.01, 0.1, 1, 10, 100, 1000]}
elif args.model == "lmm":
    params = {"lambda": [0, 1, 10, 100], "gamma": [0.01, 0.1, 1], "sigma": [0.25, 0.5, 1]}

print("----------------------------------------")
print("Dataset: %s" % args.dataset)
print("Model: %s" % args.model)
print("Loss function: %s" % args.loss)
print("Params: %s" % params)
print("n_splits: %s" % args.n_splits)
print("validation_size: %s" % args.validation_size)
print("splitter: %s" % args.splitter)
print("Execution: %s" % args.execution)
print("----------------------------------------\n")

X, bags, y, train_index, test_index = load_dataset(args)

scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(X)

X_train, y_train, bags_train = X[train_index], y[train_index], bags[train_index]
X_test, y_test, bags_test = X[test_index], y[test_index], bags[test_index]
proportions = compute_proportions(bags_train, y_train)

df_results = pd.DataFrame(columns=["accuracy_test", "best_hyperparams"])

print("Execution started!!!")

if args.model == "llp-svm-lin":
    model = alterSVM(llp_loss_function_type=args.loss, random_state=seed[args.execution])
elif args.model == "kdd-lr":
    model = EM(LogisticRegression(solver='lbfgs'), init_y="random", random_state=seed[args.execution])
elif args.model == "lmm":
    model = LMM(lmd=1, gamma=1, sigma=1, similarity="G,s")

gs = gridSearchCV(model, params, refit=True, cv=args.n_splits, splitter=args.splitter, loss_type=args.loss, 
                        validation_size=args.validation_size, central_tendency_metric="mean", 
                        n_jobs=N_JOBS, random_state=seed[args.execution])

gs.fit(X_train, bags_train, proportions, y_train)
y_pred_test = gs.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
best_hyperparams = gs.best_params_

df_results = pd.concat([df_results, pd.DataFrame([[accuracy_test, best_hyperparams]], columns=["accuracy_test", "best_hyperparams"])], ignore_index=True)
df_results.to_parquet(filename)
print("Execution finished!!!")
