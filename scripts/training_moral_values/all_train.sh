#!/bin/bash

# This script trains model on the Moral Values dataset.
# deberta models
python3 scripts/training_moral_values/train_paper.py valuenet --use_model microsoft/deberta-v3-base --n_runs 5
python3 scripts/training_moral_values/train_paper.py valueeval --use_model microsoft/deberta-v3-base --n_runs 5
python3 scripts/training_moral_values/train_paper.py both --use_model microsoft/deberta-v3-base --n_runs 5

# RoBERTa models
python3 scripts/training_moral_values/train_paper.py valuenet --use_model roberta-base --n_runs 5
python3 scripts/training_moral_values/train_paper.py valueeval --use_model roberta-base --n_runs 5
python3 scripts/training_moral_values/train_paper.py both --use_model roberta-base --n_runs 5

# BERT models
python3 scripts/training_moral_values/train_paper.py valuenet --use_model bert-base-uncased --n_runs 5
python3 scripts/training_moral_values/train_paper.py valueeval --use_model bert-base-uncased --n_runs 5
python3 scripts/training_moral_values/train_paper.py both --use_model bert-base-uncased --n_runs 5
