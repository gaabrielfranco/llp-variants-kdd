# EM/LR and LMM experiments

# for dataset in "adult-hard-10bags-cluster-kmeans-10" "adult-intermediate-10bags-cluster-kmeans-10" "adult-naive-10bags-cluster-None-None" "adult-simple-10bags-cluster-None-None" "covertype-classes-1-2-hard-10bags-cluster-kmeans-10" "covertype-classes-1-2-intermediate-10bags-cluster-kmeans-10" "covertype-classes-1-2-naive-10bags-cluster-None-None" "covertype-classes-1-2-simple-10bags-cluster-None-None"
# do
#     for model in "kdd-lr" "lmm"
#     do
#         for n_splits in "10"
#         do
#             for splitter in "full-bag-stratified-k-fold" "split-bag-k-fold"
#             do
#                 for ((exec=0; exec<30; exec++))
#                 do
#                     python3 kdd_experiment.py -d $dataset -m $model -l "abs" -n $n_splits -s $splitter -e $exec
#                 done
#             done
#             for splitter in "split-bag-bootstrap" "split-bag-shuffle"
#             do
#                 for ((exec=0; exec<30; exec++))
#                 do
#                     python3 kdd_experiment.py -d $dataset -m $model -l "abs" -n $n_splits -s $splitter -v 0.5 -e $exec
#                 done
#             done
#         done
#     done
# done

# for dataset in "default-credit-card-hard-5bags-cluster-kmeans-5" "default-credit-card-intermediate-5bags-cluster-kmeans-5" "default-credit-card-naive-5bags-cluster-None-None" "default-credit-card-simple-5bags-cluster-None-None"
# do
#     for model in "kdd-lr" "lmm"
#     do
#         for n_splits in "5"
#         do
#             for splitter in "full-bag-stratified-k-fold" "split-bag-k-fold"
#             do
#                 for ((exec=0; exec<30; exec++))
#                 do
#                     python3 kdd_experiment.py -d $dataset -m $model -l "abs" -n $n_splits -s $splitter -e $exec
#                 done
#             done
#             for splitter in "split-bag-bootstrap" "split-bag-shuffle"
#             do
#                 for ((exec=0; exec<30; exec++))
#                 do
#                     python3 kdd_experiment.py -d $dataset -m $model -l "abs" -n $n_splits -s $splitter -v 0.5 -e $exec
#                 done
#             done
#         done
#     done
# done

# for dataset in "census-kdd-balanced-hard-3bags-cluster-kmeans-3" "census-kdd-balanced-intermediate-3bags-cluster-kmeans-3" "census-kdd-balanced-naive-3bags-cluster-None-None" "census-kdd-balanced-simple-3bags-cluster-None-None" "cifar-10-grey-airplane-automobile-hard-3bags-cluster-kmeans-3" "cifar-10-grey-airplane-automobile-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-airplane-automobile-naive-3bags-cluster-None-None" "cifar-10-grey-airplane-automobile-simple-3bags-cluster-None-None" "cifar-10-grey-airplane-frog-hard-3bags-cluster-kmeans-3" "cifar-10-grey-airplane-frog-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-airplane-frog-naive-3bags-cluster-None-None" "cifar-10-grey-airplane-frog-simple-3bags-cluster-None-None" "cifar-10-grey-airplane-horse-hard-3bags-cluster-kmeans-3" "cifar-10-grey-airplane-horse-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-airplane-horse-naive-3bags-cluster-None-None" "cifar-10-grey-airplane-horse-simple-3bags-cluster-None-None" "cifar-10-grey-automobile-dog-hard-3bags-cluster-kmeans-3" "cifar-10-grey-automobile-dog-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-automobile-dog-naive-3bags-cluster-None-None" "cifar-10-grey-automobile-dog-simple-3bags-cluster-None-None" "cifar-10-grey-bird-cat-hard-3bags-cluster-kmeans-3" "cifar-10-grey-bird-cat-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-bird-cat-naive-3bags-cluster-None-None" "cifar-10-grey-bird-cat-simple-3bags-cluster-None-None" "cifar-10-grey-bird-ship-hard-3bags-cluster-kmeans-3" "cifar-10-grey-bird-ship-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-bird-ship-naive-3bags-cluster-None-None" "cifar-10-grey-bird-ship-simple-3bags-cluster-None-None" "cifar-10-grey-cat-horse-hard-3bags-cluster-kmeans-3" "cifar-10-grey-cat-horse-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-cat-horse-naive-3bags-cluster-None-None" "cifar-10-grey-cat-horse-simple-3bags-cluster-None-None" "cifar-10-grey-cat-truck-hard-3bags-cluster-kmeans-3" "cifar-10-grey-cat-truck-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-cat-truck-naive-3bags-cluster-None-None" "cifar-10-grey-cat-truck-simple-3bags-cluster-None-None" "cifar-10-grey-deer-ship-hard-3bags-cluster-kmeans-3" "cifar-10-grey-deer-ship-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-deer-ship-naive-3bags-cluster-None-None" "cifar-10-grey-deer-ship-simple-3bags-cluster-None-None" "cifar-10-grey-dog-frog-hard-3bags-cluster-kmeans-3" "cifar-10-grey-dog-frog-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-dog-frog-naive-3bags-cluster-None-None" "cifar-10-grey-dog-frog-simple-3bags-cluster-None-None" "cifar-10-grey-frog-ship-hard-3bags-cluster-kmeans-3" "cifar-10-grey-frog-ship-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-frog-ship-naive-3bags-cluster-None-None" "cifar-10-grey-frog-ship-simple-3bags-cluster-None-None" "cifar-10-grey-frog-truck-hard-3bags-cluster-kmeans-3" "cifar-10-grey-frog-truck-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-frog-truck-naive-3bags-cluster-None-None" "cifar-10-grey-frog-truck-simple-3bags-cluster-None-None" "mnist-digits-0-1-hard-3bags-cluster-kmeans-3" "mnist-digits-0-1-intermediate-3bags-cluster-kmeans-3" "mnist-digits-0-1-naive-3bags-cluster-None-None" "mnist-digits-0-1-simple-3bags-cluster-None-None" "mnist-digits-0-4-hard-3bags-cluster-kmeans-3" "mnist-digits-0-4-intermediate-3bags-cluster-kmeans-3" "mnist-digits-0-4-naive-3bags-cluster-None-None" "mnist-digits-0-4-simple-3bags-cluster-None-None" "mnist-digits-1-4-hard-3bags-cluster-kmeans-3" "mnist-digits-1-4-intermediate-3bags-cluster-kmeans-3" "mnist-digits-1-4-naive-3bags-cluster-None-None" "mnist-digits-1-4-simple-3bags-cluster-None-None" "mnist-digits-1-8-hard-3bags-cluster-kmeans-3" "mnist-digits-1-8-intermediate-3bags-cluster-kmeans-3" "mnist-digits-1-8-naive-3bags-cluster-None-None" "mnist-digits-1-8-simple-3bags-cluster-None-None" "mnist-digits-2-3-hard-3bags-cluster-kmeans-3" "mnist-digits-2-3-intermediate-3bags-cluster-kmeans-3" "mnist-digits-2-3-naive-3bags-cluster-None-None" "mnist-digits-2-3-simple-3bags-cluster-None-None" "mnist-digits-2-8-hard-3bags-cluster-kmeans-3" "mnist-digits-2-8-intermediate-3bags-cluster-kmeans-3" "mnist-digits-2-8-naive-3bags-cluster-None-None" "mnist-digits-2-8-simple-3bags-cluster-None-None" "mnist-digits-3-5-hard-3bags-cluster-kmeans-3" "mnist-digits-3-5-intermediate-3bags-cluster-kmeans-3" "mnist-digits-3-5-naive-3bags-cluster-None-None" "mnist-digits-3-5-simple-3bags-cluster-None-None" "mnist-digits-3-8-hard-3bags-cluster-kmeans-3" "mnist-digits-3-8-intermediate-3bags-cluster-kmeans-3" "mnist-digits-3-8-naive-3bags-cluster-None-None" "mnist-digits-3-8-simple-3bags-cluster-None-None" "mnist-digits-4-9-hard-3bags-cluster-kmeans-3" "mnist-digits-4-9-intermediate-3bags-cluster-kmeans-3" "mnist-digits-4-9-naive-3bags-cluster-None-None" "mnist-digits-4-9-simple-3bags-cluster-None-None" "mnist-digits-5-8-hard-3bags-cluster-kmeans-3" "mnist-digits-5-8-intermediate-3bags-cluster-kmeans-3" "mnist-digits-5-8-naive-3bags-cluster-None-None" "mnist-digits-5-8-simple-3bags-cluster-None-None" "mnist-digits-6-7-hard-3bags-cluster-kmeans-3" "mnist-digits-6-7-intermediate-3bags-cluster-kmeans-3" "mnist-digits-6-7-naive-3bags-cluster-None-None" "mnist-digits-6-7-simple-3bags-cluster-None-None" "mnist-digits-7-9-hard-3bags-cluster-kmeans-3" "mnist-digits-7-9-intermediate-3bags-cluster-kmeans-3" "mnist-digits-7-9-naive-3bags-cluster-None-None" "mnist-digits-7-9-simple-3bags-cluster-None-None"
# do
#     for model in "kdd-lr" "lmm"
#     do
#         for n_splits in "3"
#         do
#             for splitter in "full-bag-stratified-k-fold" "split-bag-k-fold"
#             do
#                 for ((exec=0; exec<30; exec++))
#                 do
#                     python3 kdd_experiment.py -d $dataset -m $model -l "abs" -n $n_splits -s $splitter -e $exec
#                 done
#             done
#             for splitter in "split-bag-bootstrap" "split-bag-shuffle"
#             do
#                 for ((exec=0; exec<30; exec++))
#                 do
#                     python3 kdd_experiment.py -d $dataset -m $model -l "abs" -n $n_splits -s $splitter -v 0.5 -e $exec
#                 done
#             done
#         done
#     done
# done


# Alter-SVM experiments
for dataset in "cifar-10-grey-airplane-automobile-hard-3bags-cluster-kmeans-3" "cifar-10-grey-airplane-automobile-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-airplane-automobile-simple-3bags-cluster-None-None" "cifar-10-grey-airplane-frog-hard-3bags-cluster-kmeans-3" "cifar-10-grey-airplane-frog-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-airplane-frog-simple-3bags-cluster-None-None" "cifar-10-grey-deer-ship-hard-3bags-cluster-kmeans-3" "cifar-10-grey-deer-ship-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-deer-ship-simple-3bags-cluster-None-None" "cifar-10-grey-frog-truck-hard-3bags-cluster-kmeans-3" "cifar-10-grey-frog-truck-intermediate-3bags-cluster-kmeans-3" "cifar-10-grey-frog-truck-simple-3bags-cluster-None-None" "mnist-digits-2-8-hard-3bags-cluster-kmeans-3" "mnist-digits-2-8-intermediate-3bags-cluster-kmeans-3" "mnist-digits-2-8-simple-3bags-cluster-None-None" "mnist-digits-4-9-hard-3bags-cluster-kmeans-3" "mnist-digits-4-9-intermediate-3bags-cluster-kmeans-3" "mnist-digits-4-9-simple-3bags-cluster-None-None" "mnist-digits-5-8-hard-3bags-cluster-kmeans-3" "mnist-digits-5-8-intermediate-3bags-cluster-kmeans-3" "mnist-digits-5-8-simple-3bags-cluster-None-None" "mnist-digits-7-9-hard-3bags-cluster-kmeans-3" "mnist-digits-7-9-intermediate-3bags-cluster-kmeans-3" "mnist-digits-7-9-simple-3bags-cluster-None-None"
do
    for model in "llp-svm-lin"
    do
        for n_splits in "3"
        do
            for splitter in "full-bag-stratified-k-fold"
            do
                for ((exec=0; exec<10; exec++))
                do
                    python3 kdd_experiment.py -d $dataset -m $model -l "abs" -n $n_splits -s $splitter -e $exec
                done
            done
            for splitter in "split-bag-bootstrap" "split-bag-shuffle"
            do
                for ((exec=0; exec<10; exec++))
                do
                    python3 kdd_experiment.py -d $dataset -m $model -l "abs" -n $n_splits -s $splitter -v 0.5 -e $exec
                done
            done
        done
    done
done