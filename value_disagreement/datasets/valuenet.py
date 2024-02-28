from pathlib import Path

import numpy as np
import pandas as pd

from value_disagreement.datasets import ValueDataset


class ValueNetDataset(ValueDataset):
    def __init__(self, data_dir, return_predefined_splits=False, focus_value=None):
        """
        Load data used in the ValueNet paper [1].

        [1] https://arxiv.org/pdf/2112.06346.pdf
        """
        self.return_predefined_splits = return_predefined_splits
        self.focus_value = focus_value

        self.dataset_dir = Path(data_dir)
        self.train_df = pd.read_csv(self.dataset_dir / "train.csv")
        self.val_df = pd.read_csv(self.dataset_dir / "eval.csv")
        self.test_df = pd.read_csv(self.dataset_dir / "test.csv")
        self.reformat_data()

    def reformat_data(self):
        """
        Recast the data from:
            [value] {text}
        to a pandas dataframe with columns:
            text, value, orig_label
        """
        new_dfs = []
        for df in [self.train_df, self.val_df, self.test_df]:
            df = df.rename(columns={'label': 'orig_label'})
            texts = []
            df_labels = []
            values = []
            for _, row in df.iterrows():
                text = row['scenario']
                value, sentence = text.split("]", 1)
                value = value.replace("[", "").lower()
                values.append(value)
                label = row['orig_label']
                new_class_label = f"{value}{label}"
                df_labels.append(new_class_label)
                texts.append(sentence)
            df['text'] = texts
            df['new_class_label'] = df_labels
            df['value'] = values
            new_dfs.append(df)

        if self.focus_value is not None:
            total_df = pd.concat(new_dfs)
            self.all_data_df = total_df[total_df['value'] == self.focus_value].copy()
            self.all_data_df['focus_value_present'] = self.all_data_df['orig_label'].apply(lambda x: np.abs(x))
        else:
            self.train_df, self.val_df, self.test_df = new_dfs
            self.values = set(values)
            self.all_data_df = pd.concat([self.train_df, self.val_df, self.test_df])
            self.all_data_df.reset_index(drop=True, inplace=True)

    def get_splits(self, train_size=0.8, val_size=0.1, test_size=0.1):
        if self.return_predefined_splits:
            train_idx = np.arange(start=0, stop=len(self.train_df))
            val_idx = np.arange(start=len(self.train_df), stop=len(self.train_df)+len(self.val_df))
            test_idx = np.arange(start=len(self.train_df)+len(self.val_df), stop=len(self.train_df)+len(self.val_df)+len(self.test_df))
            return train_idx, val_idx, test_idx
        else:
            all_idx = np.arange(0, len(self.all_data_df))
            np.random.shuffle(all_idx)
            train_idx = all_idx[:int(len(self.all_data_df)*train_size)]
            val_idx = all_idx[int(len(self.all_data_df)*train_size):int(len(self.all_data_df)*(train_size+val_size))]
            test_idx = all_idx[int(len(self.all_data_df)*(train_size+val_size)):]
            return train_idx, val_idx, test_idx


    def __len__(self):
        return len(self.all_data_df)

    def __getitem__(self, idx):
        if self.focus_value:
            return self.all_data_df.iloc[idx]['text'], self.all_data_df.iloc[idx]['focus_value_present']
        else:
            return self.all_data_df.iloc[idx]
