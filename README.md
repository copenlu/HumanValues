# Human Values

Repository of the paper ["Investigating Human Values in Online Communities"](https://arxiv.org/abs/2402.14177)

Still a work in progress.

1. The list of subreddit we analised can be found in `outputs/subreddits.txt`.
2. Subreddit Schwartz values can be found in `outputs/subreddit_schwartz_values.csv`.
3. Figures from the paper can be found in `outputs/`.

## Usage
To use the code, clone it to a new folder, cd into it, and run `conda env create -f environment.yml`, which will create a fresh conda enviroment with all the needed dependencies.

### Data Collection
First, download the posts data from [pushshift.io](https://pushshift.io/) (the `RS_*.zst` files). In the paper, we analised the files `RS_2022-01.zst, ..., RS_2022-09.zst`, but any set of `RS_*.zst` files will do. These files are very large (compresed size can surpass 15GB, and the uncompressed size can surpass 200GB), so save them in a system with enough storage.

Next, run the script 

```bash
python collect_reddit_data_to_label.py [PATH TO PUSHIFT] [SAVE PATH]
```

It will collect up to 1000 random posts from each subreddit in `outputs/subreddits.txt` and store them in `[SAVE PATH]`.

### Training

We use the training code from `https://github.com/m0re4u/value-disagreement`, the repositopry of the paper "*Do Differences in Values Influence Disagreements in Online Discussions?*".
Either use their repository to train the model, or follow their instructions while using ours, which contains a simplified, slim, version of their code. The training script is `scripts/training_moral_values/train_paper.py`. To use it, run

```bash
python scripts/training_moral_values/train_paper.py both --use_model microsoft/deberta-v3-base --n_runs 1
```

Do not forget to download `ValueEval` and `ValueNet` datasets and store them in `data` directory.

### Labeling subreddits

To label the posts collected by `collect_reddit_data_to_label.py` with Schwartz values, run

```bash
python label_subreddits.py [POSTS PATH] [MODEL CHECKPOINT] [SAVE DIR]
```

Where `[POSTS PATH]` is the same as `[SAVE PATH]` from the data collection step, `[MODEL CHECKPOINT]` is the checkpoint of the trained model from the training step. The code also supports basic parrallelism -- you can include the arguments `--subset [SUBSET ID]` and `--total [N SUBSETS]` to process only subset number `[SUBSET ID]` out of `[N SUBSETS]`.


### Data analysis

TBD. The notebook `analyse_subreddits.ipynb` contains some reference code.

