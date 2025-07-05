#!/bin/bash

# tmp dir for matplotlib
export MPLCONFIGDIR="/tmp/matplotlib-$USER"
mkdir -p "$MPLCONFIGDIR"

export PYTHONPATH="/sac/src:$PYTHONPATH"

cd /sac/src



# ---------------------
# HPO for PD with Transformer Encoder
# ---------------------
# for i in {1..4}
# do
#   echo "Running HPO for Outer-fold: $i"
#   python 01_optimize_hpo.py \
#     --condition PD \
#     --aggregator transformer_encoder \
#     --fold_index $i \
#     --n_trials 300
#   sleep 300 # cool-down
# done
# exit

# ---------------------
# run analysis scripts (commented out)
# ---------------------
python 01_train.py