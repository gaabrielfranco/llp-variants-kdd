# Dependence and Model Selection in LLP: The Problem of Variants
Repository for reproducing the results of the paper "Dependence and Model Selection in LLP: The Problem of Variants" from KDD23

The Split-Bag methods proposed at this paper are implemented at [llp-learn](https://github.com/gaabrielfranco/llp-learn)

## Requirements
- Python 3.8 or higher (developed on Python 3.8)
- R version 4.2.1

```sh
pip3 install -r requirements.txt
```

To use LMM[^1] it is necessary to get its code:

```sh
git clone https://github.com/giorgiop/almostnolabel.git
```

[^1]: Patrini, Giorgio, et al. "(Almost) no label no cry." Advances in Neural Information Processing Systems 27 (2014).

To install the R libraries:
```sh
install_r_libraries.py
```

## Run an single experiment

```sh
python3 kdd_experiment.py -d {dataset_name} -m {model} -l {loss} -n {n_splits} -v {validation_size_percentage} -s {splitter} -e {execution_number}
```

As an example, we have:
```sh
python3 kdd_experiment.py -d mnist-digits-6-7-naive-3bags-cluster-None-None -m lmm -l abs -n 3 -v 0.5 -s split-bag-bootstrap -e 0
```

For $k$-fold based methods, the *validation_size_percentage* is not used
```sh
python3 kdd_experiment.py -d mnist-digits-6-7-naive-3bags-cluster-None-None -m lmm -l abs -n 3 -s split-bag-k-fold -e 0
```

## Run all the paper experiments

```sh
./paper_run_all_experiments.sh
```

Each execution produces one ```parquet``` file. After running all the experiments, they can be combined into one single file (```kdd-experiment-results.parquet```) as following:

```sh
python3 aggregate_results.py
```

## Produce all the plots in the paper

```sh
./paper_plot_results.sh
```

The plots are saved in the ```plots``` folder.

## Produce the results and extra information about the datasets in LaTeX table format
```sh
./paper_table_results.sh
```

The tables are saved in the ```tables``` folder.
