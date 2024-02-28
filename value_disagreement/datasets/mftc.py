import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from value_disagreement.extraction import ValueConstants


class MFTCDataset():
    def __init__(self):
        self.dataset_dir = Path("data/MFTC/raw/MFTC_V4_text.json")
        with open(self.dataset_dir) as f:
            self.data = json.load(f)

    def get_label_mapping(self):
        label_set = set(ValueConstants.MFT_VALUES)
        id2label = {i:l for i, l in enumerate(label_set)}
        label2id = {v:k for k, v in id2label.items()}
        return id2label, label2id

    def get_label(self, annotations, label2id):
        assigned_labels = []
        assigned_labels_orig = []
        num_annotators = len(set([x['annotator'] for x in annotations]))

        label_counter = defaultdict(lambda: [])
        for ann in annotations:
            labels = ann['annotation'].split(',')
            for label in labels:
                if label == "nm" or label == "nh":
                    label = "non-moral"
                label_counter[label].append(ann['annotator'])

        majority_threshold = num_annotators / 2
        for label in label_counter:
            if len(set(label_counter[label])) >= majority_threshold:
                assigned_labels.append(float(label2id[label]))
                assigned_labels_orig.append([label])

        if len(assigned_labels) == 0:
            assigned_labels.append(float(label2id["non-moral"]))
            assigned_labels_orig.append(["non-moral"])
        return assigned_labels, assigned_labels_orig


    def json_subcorpus_to_df(self, corpus, label2id):
        rows = []
        for item in corpus['Tweets']:
            labels, labels_orig = self.get_label(item['annotations'], label2id)
            row = {
                'text': item['tweet_text'],
                'labels': labels,
                'labels_orig': labels_orig,
            }
            rows.append(row)
        df = pd.DataFrame.from_records(rows)
        return df

    def prepare_datasets(self, label2id):
        ALL_SETS = [0,1,2,3,4,5,6]

        dfs = []
        for subreddit_set in ALL_SETS:
            corpus = self.data[subreddit_set]
            subreddit_df = self.json_subcorpus_to_df(corpus, label2id)
            dfs.append(subreddit_df)

        return pd.concat(dfs)

    def reformat_data(self):
        self.id2label, self.label2id = self.get_label_mapping()
        self.all_df = self.prepare_datasets(self.label2id)

    def get_splits(self, train_size=0.8, val_size=0.1, test_size=0.1):
        assert train_size + val_size + test_size == 1
        num_train = int(len(self)*train_size)
        num_val = int(len(self)*val_size)
        idx = list(range(len(self)))
        np.random.shuffle(idx)
        train_set = idx[:num_train]
        val_set = idx[num_train:num_train + num_val]
        test_set = idx[num_train + num_val:]
        return self.all_df.iloc[:train_set], self.all_df.iloc[val_set], self.all_df.iloc[test_set]


    def __len__(self):
        return len(self.all_df)
