from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset as TorchDataset


class DebagreementDataset(TorchDataset):
    def __init__(self, path_or_df, sample_N=-1, subreddit=None):
        if isinstance(path_or_df, str) or isinstance(path_or_df, Path):
            self.df = pd.read_csv(path_or_df)
        else:
            self.df = path_or_df

        if sample_N > 0:
            self.df = self.df.sample(sample_N)

        if subreddit is not None:
            self.subreddit = subreddit
            self.df = self.df[self.df.subreddit.str.lower() == subreddit.lower()]


        self.id2label = {0: "disagree", 1: "neutral", 2: "agree"}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return body text of the parent post, body text of the child post, and the label.

        Also return the author usernames of parent and child, and subreddit information.
        """
        row = self.df.iloc[idx]
        extra_info_dict = {
            'author_parent': row.author_parent,
            'author_child': row.author_child,
            'subreddit': row.subreddit,
            'submission_text': row.submission_text
        }
        if 'sim_scores' in self.df.columns:
            extra_info_dict['sim_scores'] = row.sim_scores
        return row.body_parent, row.body_child, row.label, extra_info_dict

    def get_splits(self, df_group, train_len_pct=0.80, val_len_pct=0.10):
        """
        Split the data into train, val, and test sets.
        """
        total_length = len(df_group)
        train_length = int(train_len_pct * total_length)
        val_length = int(val_len_pct * total_length)
        test_length = total_length - (train_length + val_length)
        train_set = df_group.iloc[:train_length]
        val_set = df_group.iloc[train_length:train_length + val_length]
        test_set = df_group.iloc[train_length + val_length:]
        assert len(train_set) == train_length
        assert len(val_set) == val_length
        assert len(test_set) == test_length
        return train_set, val_set, test_set

    def get_ordered_splits(self):
        """
        Split the data into train, val, and test sets ordered by the timestamp.
        """
        trains = []
        vals = []
        tests = []
        for _, g in self.df.groupby("subreddit"):
            g["date_py"] = pd.to_datetime(self.df["datetime"])
            g = g.sort_values(by="date_py")
            train, val, test = self.get_splits(g)
            trains.append(train)
            vals.append(val)
            tests.append(test)

        df_train = pd.concat(trains)
        df_val = pd.concat(vals)
        df_test = pd.concat(tests)
        return DebagreementDataset(df_train), DebagreementDataset(df_val), DebagreementDataset(df_test)

    def filter_authors(self, author_list):
        """
        Filter the dataset to only include posts from the given list of authors.
        """
        parent = self.df.author_parent.isin(author_list)
        child = self.df.author_child.isin(author_list)
        self.df = self.df[parent & child]

    def compute_profile_similarities(self, user2vectors, scaling="standard"):
        """
        Compute the cosine similarity between the user profiles of the parent and child posts.
        """
        parent_vectors = np.array([x for x in self.df.author_parent.map(user2vectors)])
        child_vectors = np.array([x for x in self.df.author_child.map(user2vectors)])
        sim_scores = cosine_similarity(parent_vectors, child_vectors).diagonal()
        # Optionally, scale the sim scores
        if scaling == "standard":
            scaler = StandardScaler()
            in_X = sim_scores.reshape(-1, 1)
            sim_scores = scaler.fit_transform(in_X)
        # Round sim score for tokens
        rounded_scores = np.around(sim_scores * 100, decimals=0)
        rounded_scores = np.clip(rounded_scores, -100, 25)
        self.df['sim_scores'] = rounded_scores


    @staticmethod
    def create_hf_dataset(docs, tokenize_function):
        """
        Convert the dataset into a format that can be used by HuggingFace models.
        """
        dataset = Dataset.from_dict({'text': docs})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset, np.ones((len(docs),))
