# Human Values

Repository of the paper ["Investigating Human Values in Online Communities"](https://arxiv.org/abs/2402.14177)

Still a work in progress.

1. Subreddit Schwartz values can be found in `outputs/subreddit_schwartz_values.csv`.
2. Figures from the paper can be found in `outputs/`

## Usage
To use the code, clone it to a new folder, cd into it, and run `conda env create -f environment.yml`, which will create a fresh conda enviroment with all the needed dependencies.

### Data Collection
First, download the posts data from `https://pushshift.io/` (the `RS_*.zst` files). In the paper, we analised the files `RS_2022-01.zst, ..., RS_2022-09.zst`, but any set of `RS_*.zst` files will do. The files are very large (compresed size can surpass 15GB, and the uncompressed size can surpass 200GB), so save them in a system with enough storage.



### Training

We use the training code from `https://github.com/m0re4u/value-disagreement`, the repositopry of the paper "Do Differences in Values Influence Disagreements in Online Discussions?".
Either use their repository to train the model, or follow their instructions while using ours, which contains a simplified, slim, version of their code. The training script is `scripts/training_moral_values/train_paper.py`.

### Data analysis

TBD

