# MulCo-legacy
# This directory contains the source code of MulCo implementation (alpha version), published in Conference on Information and Knowledge Management (read paper at https://dl.acm.org/doi/abs/10.1145/3627673.3679520).

# This research was supported in part by the Intramural Research Program of the National Institutes of Health (NIH). The contributions of the NIH author(s) were made as part of their official duties as NIH federal employees, are in compliance with agency policy requirements, and are considered Works of the United States Government. However, the findings and conclusions presented in this paper are those of the author(s) and do not necessarily reflect the views of the NIH or the U.S. Department of Health and Human Services.

# Required python package:
# 1. torch
# 2. transformers
# 3. numpy
# 4. networkx
# 5. tqdm
# 6. spacy
# Please install all of them via conda manager.

# This is pure CUDA implementation. To run on CPU, simply remove all .cuda() statements for all variables.
# Be aware, this is running on DGX-A100 8-GPU Server. As we tested, each single GPU runs differently. However, we do our best to select seeds that are stable to reproduce the experiment.
# Adjusting the gradient accumulation_steps argument if GPU memory is not sufficient. Generally, it requires at least 24GB GPU memory to run.

# The source code has been tested and successfully run on the following GPUs: 
# 1. GTX 1080 Ti - 11GB (Require very small batch size with accumulation_steps)
# 2. RTX 2080 Ti - 11GB (Require very small batch size with accumulation_steps)
# 3. RTX 3090 - 24GB
# 4. RTX A6000 - 48 GB
# 5. DGX-Station Tesla V100 - 32GB
# 6. DGX-A100 A100 - 40GB

# MulCo-formatted data in this repository taken from public datasets TimeBankDense (https://www.usna.edu/Users/cs/nchamber/caevo/#corpus), TDDiscourse (https://github.com/aakanksha19/TDDiscourse), and MATRES (https://github.com/qiangning/MATRES). See "data_formatting.txt" for instructions on formatting new data sources for MulCo.

# Sample running commands
# TDDMan
python3 MulCo_pipeline.py --doc_file data/processed_docs.pkl --gold_pairs data/tdd_man_event_pairs.pkl --test_pairs data/tdd_test_pair_list.pkl --out_dir models/tdd_man --learning_rate 1e-5 --batch_size 16 --gcn_type rgat --gcn_num_layers 2 --accumulation_steps 0 --seed 1103 --gcn_hidden_size 256 --epochs 5 --k_hops 1 --bert_model roberta --dropout 0.1 --temp 0.1  

# TDDAuto
python3 MulCo_pipeline.py --doc_file data/processed_docs.pkl --gold_pairs data/tdd_auto_event_pairs.pkl --test_pairs data/tdd_test_pair_list.pkl --out_dir models/tdd_auto --learning_rate 1e-5 --batch_size 32 --gcn_type rgat --gcn_num_layers 3 --accumulation_steps 8 --seed 2513 --gcn_hidden_size 256 --epochs 8 --k_hops 2 --dropout 0.1 --bert_model roberta --temp 0.04 --multi_scale

# MATRES
python MulCo_pipeline.py --doc_file data/processed_docs.pkl --gold_pairs data/matres_event_pairs.pkl --test_pairs data/matres_event_pairs.pkl --out_dir models/matres --learning_rate 1e-5 --batch_size 16 --gcn_type rgat --gcn_num_layers 1 --accumulation_steps 0 --seed 1103 --gcn_hidden_size 128 --epochs 10 --k_hops 1 --bert_model roberta --dropout 0.3  --temp 0.9

# TB-Dense (Timebank-Dense)
python3 MulCo_pipeline.py --doc_file data/processed_docs.pkl --gold_pairs data/tbd_event_pairs.pkl --test_pairs data/tbd_test_pair_list.pkl --out_dir models/tbd --learning_rate 1e-5 --batch_size 32 --gcn_type rgat --gcn_num_layers 1 --k_hops 1 --accumulation_steps 0 --seed 2513 --gcn_hidden_size 256 --epochs 10 --dropout 0.1 --bert_model roberta --temp 0.9 --multi_scale