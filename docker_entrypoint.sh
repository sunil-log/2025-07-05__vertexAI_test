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
python 04_analyze_best_trials.py \
	--base_dir "trials/250704-0736__pd_transformer_encoder__transformer__L2_added" \
	--top_n 50 \
	--param_filter "tf_dropout:<:1.0"
exit

# python 05_analyze_all_outer_folds.py \
#     --base_dir "trials/250629-1641__PD_HPO_all__good_score" \
#     --top_n 100
