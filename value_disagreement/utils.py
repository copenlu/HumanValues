import os
import random
from functools import lru_cache

import matplotlib.pyplot as plt
import mong
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=1000000000)
def lemmatize(w: str):
    """
    Caching the word-based lemmatizer to speed the process up.
    """
    return lemmatizer.lemmatize(w)


def lemmatize_dataset(dataset):
    """
    Lemmatize all words in a dataset and only keep alphanumerical characters
    """
    lemmatized_sentences = []
    stops = set(stopwords.words("english"))
    for sentence in dataset:
        lem_sent = [lemmatize(w.word.lower()) for w in sentence.tokens if not lemmatize(w.word.lower()) in stops]
        lemmatized_sentences.append(lem_sent)
    return lemmatized_sentences


def plot_sentence_lengths(data, fname, count_target='tokens'):
    """
    Plot the length of the sentences in the dataset.

    Length in terms of `tokens` or `characters`
    """
    if count_target not in ['tokens', 'characters']:
        raise ValueError("Can only plot lengths of `tokens` and `characters`")

    lens = []
    for sentence in data:
        if count_target == 'tokens':
            lens.append(len(sentence))
        elif count_target == 'characters':
            lens.append(len(sentence.get_text()))
    ax = sns.displot(lens)
    ax.set(xlabel=f'input sentence {count_target} length')
    plt.savefig(fname)

def dump_sentences(data, fname):
    """
    Dump the sentences in a given dataset to txt file.
    """
    with open(fname, 'w') as f:
        for i in range(len(data)):
            print_string = data[i].get_text().replace('\n', '')
            f.write(f"{print_string}\n")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wandb_get_name():
    """
    Get the name of the run for wandb.
    """
    return mong.get_random_name()

def print_stats(result_type, stats_list):
    """
    Print average and standard deviation of a list of numbers.
    """
    print(f"{result_type}: {np.average(stats_list):.2f} ({np.std(stats_list):.2f}) ")
