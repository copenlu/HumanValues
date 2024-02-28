from pathlib import Path

import numpy as np
import pandas as pd

from value_disagreement.datasets import ValueDataset
from value_disagreement.extraction.moral_values import ValueConstants


class ValueEvalDataset(ValueDataset):
    def __init__(
        self,
        data_dir,
        focus_value=None,
        cast_to_valuenet=False,
        return_predefined_splits=False,
    ):
        """
        Load data from the Value Argument paper [1].

        [1] https://aclanthology.org/2022.acl-long.306.pdf
        """
        self.dataset_dir = Path(data_dir)
        self.arguments = pd.read_csv(self.dataset_dir / "arguments.tsv", sep="\t")
        self.labels = pd.read_csv(self.dataset_dir / "labels-level2.tsv", sep="\t")
        self.label_df = self.labels.iloc[:, 1:]
        self.arguments["raw_labels"] = [
            self.label_df.columns[i].tolist()
            for i in self.label_df.values
            == np.array(self.label_df.max(axis=1))[:, None]
        ]

        # Remapping of the labels
        self.label_mapping = {
            "Humility": None,
            "Universalism: objectivity": ValueConstants.SCHWARTZ_VALUES[9],
            "Hedonism": ValueConstants.SCHWARTZ_VALUES[3],
            "Universalism: nature": ValueConstants.SCHWARTZ_VALUES[9],
            "Stimulation": ValueConstants.SCHWARTZ_VALUES[7],
            "Self-direction: action": ValueConstants.SCHWARTZ_VALUES[6],
            "Achievement": ValueConstants.SCHWARTZ_VALUES[0],
            "Benevolence: caring": ValueConstants.SCHWARTZ_VALUES[1],
            "Power: resources": ValueConstants.SCHWARTZ_VALUES[4],
            "Self-direction: thought": ValueConstants.SCHWARTZ_VALUES[6],
            "Conformity: interpersonal": ValueConstants.SCHWARTZ_VALUES[2],
            "Power: dominance": ValueConstants.SCHWARTZ_VALUES[4],
            "Conformity: rules": ValueConstants.SCHWARTZ_VALUES[2],
            "Security: societal": ValueConstants.SCHWARTZ_VALUES[5],
            "Benevolence: dependability": ValueConstants.SCHWARTZ_VALUES[1],
            "Universalism: concern": ValueConstants.SCHWARTZ_VALUES[9],
            "Face": None,
            "Universalism: tolerance": ValueConstants.SCHWARTZ_VALUES[9],
            "Security: personal": ValueConstants.SCHWARTZ_VALUES[6],
            "Tradition": ValueConstants.SCHWARTZ_VALUES[8],
        }
        self.arguments["text"] = self.arguments[["Conclusion", "Premise"]].apply(
            self.remap_text, axis=1
        )
        self.arguments["schwartz_labels"] = self.arguments["raw_labels"].apply(
            self.remap_labels
        )
        self.focus_value = focus_value
        self.return_predefined_splits = return_predefined_splits
        self.valuenet_format = False
        if focus_value is not None:
            self.reformat_label()
        if cast_to_valuenet:
            self.cast_to_valuenet()
            self.valuenet_format = True

    def reformat_label(self):
        """
        Reformat labels to be binary (0/1) for the focus value.
        """
        self.arguments["focus_value_present"] = self.arguments["schwartz_labels"].apply(
            lambda x: int(self.focus_value in x)
        )

    def cast_to_valuenet(self):
        """
        Cast to the same format as the ValueNet dataset.

        Needs to contain:
            text, value, orig_label
        """
        all_samples = []
        for i, row in self.arguments.iterrows():
            sch_labels = set(row["schwartz_labels"])
            for value in row["schwartz_labels"]:
                all_samples.append((row["Premise"], 1, value, row["Usage"]))
            not_active_sch = set(ValueConstants.SCHWARTZ_VALUES) - sch_labels
            for value in not_active_sch:
                all_samples.append((row["Premise"], 0, value, row["Usage"]))

        self.arguments = pd.DataFrame.from_records(
            all_samples, columns=["text", "orig_label", "value", "split"]
        )

    def remap_text(self, row):
        return f"{row['Conclusion']}. {row['Premise']}."

    def remap_labels(self, row):
        return [self.label_mapping[x] for x in row if self.label_mapping[x] is not None]

    def get_splits(self, train_size=0.8, val_size=0.1, test_size=0.1):
        assert train_size + val_size + test_size == 1
        num_train = int(len(self) * train_size)
        num_val = int(len(self) * val_size)
        num_test = len(self) - (num_train + num_val)

        if self.return_predefined_splits:
            # load the indices from the splits.npy
            load_splits_path = Path(f"{self.dataset_dir}/splits.npy")
            if load_splits_path.exists():
                test_set_index = np.load(load_splits_path, allow_pickle=True)
            else:
                samples_per_val = np.round(
                    num_test
                    * np.random.dirichlet(
                        np.ones(len(ValueConstants.SCHWARTZ_VALUES)), size=1
                    )
                )
                # sample a balanced test set
                test_set = []
                for i, val in enumerate(ValueConstants.SCHWARTZ_VALUES):
                    this_val_df = self.arguments[self.arguments["value"] == val]
                    val_test_df = this_val_df.sample(n=int(samples_per_val[0, i]))
                    test_set.append(val_test_df)

                test_set = pd.concat(test_set)
                test_set_index = np.array(test_set.index)
                with open(load_splits_path, "wb") as f:
                    np.save(f, test_set_index)
            # the rest is used in learning
            left_set = self.arguments[~self.arguments.index.isin(test_set_index)]
            # split rest into validation and training
            train_set = left_set[:num_train]
            val_set = self.arguments[
                ~self.arguments.index.isin(test_set_index)
                & ~self.arguments.index.isin(train_set.index)
            ]
            return np.array(train_set.index), np.array(val_set.index), test_set_index
        else:
            idx = np.arange(len(self))
            np.random.shuffle(idx)
            train_set = idx[:num_train]
            val_set = idx[num_train : num_train + num_val]
            test_set = idx[num_train + num_val :]
            # potentially you can save the split indices here to create splits.npy
            return train_set, val_set, test_set

    def __len__(self):
        return len(self.arguments)

    def __getitem__(self, idx):
        if self.valuenet_format:
            return self.arguments.iloc[idx]
        if self.focus_value is not None:
            return (
                self.arguments.iloc[idx]["text"],
                self.arguments.iloc[idx]["focus_value_present"],
            )
        else:
            text = self.arguments["Premise"].iloc[idx]
            labels = self.arguments["schwartz_labels"].iloc[idx]
            return text, labels
