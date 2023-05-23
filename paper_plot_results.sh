for plot_type in "effect-sizes" "hyperparams-plots" "winning-figures"
do
    python3 plot_results.py -p $plot_type
done