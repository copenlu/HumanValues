import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from value_disagreement.datasets import ValueDataset
from value_disagreement.extraction import ValueConstants


class RedditBackgroundDataset():
    def __init__(self, subreddit, load_csv=None):
        self.subreddit = subreddit
        if load_csv is None:
            self.dataset_dir = f"output/reddit_data_{subreddit}/*.json"
            self.files = glob.glob(self.dataset_dir)
            self.user2data, self.user2metadata = self.load_users()
        else:
            self.df = pd.read_csv(load_csv, low_memory=False, lineterminator='\n', index_col=0)
            self.user2data = self.df.groupby('user')['text'].apply(list).to_dict()
            self.user2data = {k:[str(x) for x in v] for k,v in self.user2data.items()}

    def load_users(self):
        """
        Load user background data from JSON files.
        """
        user_data = {}
        user_metadata = {}
        for userfile in self.files:
            username = Path(userfile).stem
            with open(userfile) as f:
                data = json.load(f)
            user_data[username] = [x['body'] for x in data]
            user_metadata[username] = [x for x in data]
        return user_data, user_metadata

    def __len__(self):
        """
        Return the number of comments in the dataset
        """
        return sum([len(x) for _, x in self.user2data.items()])

    def get_values_df(self):
        """
        Extract a pandas DataFrame with rows for every comment/value pair.
        """
        entries = []
        for user, comments in self.user2data.items():
            for comment in comments:
                for value in ValueConstants.SCHWARTZ_VALUES:
                    entries.append({
                        "text": comment,
                        "value": value,
                        "user": user,
                        "scenario": f"[{value.upper()}] {comment}"
                    })

        return pd.DataFrame.from_records(entries)


class RedditAnnotatedDataset(ValueDataset):
    def __init__(self, load_path, focus_value=None):
        self.label2id = {x: i for i, x in enumerate(ValueConstants.SCHWARTZ_VALUES)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        self.annotation_file = load_path
        self.labeled_examples = pd.read_csv(self.annotation_file, index_col=0)
        self.labeled_examples = self.labeled_examples[self.labeled_examples['annotation'].notna()]
        self.labeled_examples['labels'] = self.labeled_examples['annotation'].apply(self.map_labels)
        self.focus_value = focus_value
        if self.focus_value is not None:
            self.reformat_label()

    def reformat_label(self):
        self.labeled_examples['focus_value_present'] = self.labeled_examples['labels'].apply(lambda x: int(self.label2id[self.focus_value] in x))

    def __len__(self):
        return len(self.labeled_examples)

    def __getitem__(self, idx):
        text = self.labeled_examples['text'].iloc[idx]
        if self.focus_value is None:
            labels = self.labeled_examples['labels'].iloc[idx]
            return text, labels
        else:
            return text, self.labeled_examples['focus_value_present'].iloc[idx]

    def map_labels(self, labels):
        label_list = json.loads(str(labels).replace("'","\""))
        return [self.label2id[label] for label in label_list]

    def balance_data(self, downsample=True):
        """
        Balance the data by downsampling the majority class
        """
        label_counter = Counter(self.labeled_examples['focus_value_present'])
        mc = label_counter.most_common()
        # Probably label=0
        majority_label, _ = mc[0]
        # Probably label=1
        other_label, other_count = mc[1]

        if downsample:
            df_majority = self.labeled_examples[self.labeled_examples['focus_value_present'] == majority_label]
            df_majority_downsampled = df_majority.sample(n=other_count)
            df_other = self.labeled_examples[self.labeled_examples['focus_value_present'] == other_label]
            self.labeled_examples = pd.concat([df_majority_downsampled, df_other]).reset_index(drop=True)
        else:
            raise NotImplementedError("Upsampling of non-majority label not implemented")

    def get_splits(self, train_size=0.8, val_size=0.1, test_size=0.1):
        assert train_size + val_size + test_size == 1
        num_train = int(len(self)*train_size)
        num_val = int(len(self)*val_size)
        idx = list(range(len(self)))
        np.random.shuffle(idx)
        train_set = idx[:num_train]
        val_set = idx[num_train:num_train + num_val]
        test_set = idx[num_train + num_val:]
        return train_set, val_set, test_set
