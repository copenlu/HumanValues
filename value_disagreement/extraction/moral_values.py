import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import spacy
import torch
from datasets import Dataset, concatenate_datasets
from datasets.utils.logging import set_verbosity_error as datasets_set_verbosity_error
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.utils.logging import set_verbosity_error as tf_set_verbosity_error


class ValueTokenizer:
    def __init__(
        self, model_name, input_concat=False, label_type="copy", label2id=None
    ):
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.input_concat = input_concat
        self.label_type = label_type
        self.label2id = None

    def nhot_labels(self, batch_labels, num_classes=10):
        """
        batch_labels contains list of indices, convert to n-hot encoded matrix.
        """
        batch_size = len(batch_labels)
        labels = np.zeros((batch_size, num_classes), dtype=np.float64)
        for i, ex in enumerate(batch_labels):
            for v in ex:
                labels[i, v] = float(1.0)
        return labels

    def tokenize(self, examples):
        # Gather input text
        if self.input_concat:
            batch_size = len(examples["text"])
            batched_inputs = [
                f"<{examples['value'][i]}> "
                + f"{self.tokenizer.sep_token} "
                + examples["text"][i]
                for i in range(batch_size)
            ]
        else:
            batched_inputs = examples["text"]

        # Tokenize~!
        samples = self.tokenizer(batched_inputs, truncation=True, padding=True)

        # Gather target labels
        if self.label_type == "mftc":
            new_labels = []
            for i, labels in enumerate(examples["labels"]):
                doc_labels = [0.0] * 11
                for label in labels:
                    doc_labels[int(label)] = 1.0
                new_labels.append(doc_labels)
            samples["labels"] = new_labels
        elif self.label_type == "cast_float":
            samples["labels"] = [float(x) for x in examples["orig_label"]]
        elif self.label_type == "cast_nhot":
            samples["labels"] = self.nhot_labels(examples["labels"], num_classes=10)
        elif self.label_type == "cast_nhot_schwartz":
            batch_labels = []
            for multi_labels in examples["schwartz_labels"]:
                batch_labels.append([self.label2id[x] for x in multi_labels])
            samples["labels"] = self.nhot_labels(
                batch_labels, num_classes=len(self.label2id)
            )
        elif self.label_type == "copy":
            samples["labels"] = examples["labels"]
        return samples


class ValueExtractor:
    def __init__(self, load_path, type_name, batch_size=8):
        self.type_name = type_name
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(load_path)
        training_args = TrainingArguments(
            f"{type_name}_predictor",
            num_train_epochs=1,
            evaluation_strategy="epoch",
            save_strategy="no",
            disable_tqdm=True,
            per_gpu_eval_batch_size=batch_size,
            per_gpu_train_batch_size=batch_size,
        )
        self.trainer = Trainer(
            model,
            training_args,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        tf_set_verbosity_error()

    def tokenize(self, examples):
        raise NotImplementedError()

    def collapse_mft(self, profile):
        """
        Stack the virtues and vice value counts and sum along same foundation
        """
        if self.type_name != "mft":
            raise ValueError(
                "Model not in MFT value prediction mode, perhaps the wrong model was loaded?"
            )
        # Non-moral class was deleted, so adjust the virtues/vices indices
        if profile.shape[1] == 10:
            virtues_idx = [x if x < self.NON_MORAL_IDX else x - 1 for x in self.virtues]
            vices_idx = [x if x < self.NON_MORAL_IDX else x - 1 for x in self.vices]
            virts = profile[:, virtues_idx]
            vices = profile[:, vices_idx]
        else:
            virts = profile[:, self.virtues]
            vices = profile[:, self.vices]
        reduced_profile = np.stack([virts, vices])
        return np.sum(reduced_profile, axis=0)

    def predict_values(self, author_dataset, prob_threshold=0.5):
        """
        Predict value mentions for samples in a huggingface dataset.
        """
        raise NotImplementedError()

    def write_predictions_to_file(self, predictions, filename):
        """
        Write individual predictions to file
        """
        with open(f"output/user_profile_predictions/{filename}", "w") as f:
            json.dump(predictions, f)

    def create_user_dataset(self, user_comments, i):
        """
        Prepare text input for passing through model
        """
        raise NotImplementedError()

    def extract_profile_from_predictions(
        self, user_dataset, user_predictions, collapse_mft
    ):
        """
        Create a profile vector from prediction output from model
        """
        raise NotImplementedError()

    def profile_exists(self, user):
        """
        Check if a profile has already been created for a user
        """
        return Path(f"output/user_profile_predictions/pred_user={user}.json").exists()

    def load_profile(self, user):
        """
        If a profile has already been created, load it from file
        """
        p = Path(f"output/user_profile_predictions/pred_user={user}.json")
        label2id = {v: i for i, v in enumerate(ValueConstants.SCHWARTZ_VALUES)}
        with open(p, "r") as f:
            data = json.load(f)
        profile = np.zeros((len(ValueConstants.SCHWARTZ_VALUES),))
        for sample in data:
            for predicted_value in set(sample["values"]):
                profile[label2id[predicted_value]] += 1
        return profile

    def predict_multiple_users(self, comments_from_users):
        """
        Predict the values for comments from users with a single run.
        """
        user_datasets = []
        masks = []
        for i, user in enumerate(comments_from_users):
            ud, mask = self.create_user_dataset(comments_from_users[user], i)
            user_datasets.append(ud)
            masks.append(mask)

        # Combine datasets
        total_dataset = concatenate_datasets(user_datasets)
        total_dataset = total_dataset.map(self.tokenize, batched=True)
        dataset_mask = np.concatenate(masks)

        # Run prediction
        predictions = self.predict_values(total_dataset)
        return user_datasets, predictions, dataset_mask

    def predict_single_user(self, comments):
        """
        Predict the values for comments from a single user.
        """
        datasets_set_verbosity_error()
        user_dataset = self.create_user_dataset(comments, 0)
        user_dataset = user_dataset.map(self.tokenize, batched=True)
        predictions = self.predict_values(user_dataset, prob_threshold=0.9)
        return predictions

    def profile_multiple_users(
        self, comments_from_users, collapse_mft=False, write_preds=True
    ):
        """
        Profile users sequentially with a single run through the model.

        Try to look for already processed profiles by checking if the user has a profile file.
        """
        author2profile = {}
        for user in tqdm(comments_from_users):
            if self.profile_exists(user):
                print("Loading user profile from file...")
                profile = self.load_profile(user)
            else:
                user_predictions = self.predict_single_user(comments_from_users[user])
                profile, text2values = self.extract_profile_from_predictions(
                    comments_from_users[user], user_predictions
                )
                if write_preds:
                    self.write_predictions_to_file(
                        text2values, f"pred_user={user}.json"
                    )

            author2profile[user] = profile
        return author2profile


class ValueNetExtractor(ValueExtractor):
    def __init__(self, load_path, batch_size=8):
        super().__init__(load_path, "schwartz", batch_size=batch_size)
        self.label_mapping = {
            i: v for i, v in enumerate(ValueConstants.SCHWARTZ_VALUES)
        }
        self.values = ValueConstants.SCHWARTZ_VALUES

    def tokenize(self, examples):
        batch_size = len(examples["text"])
        batched_inputs = [
            f"<{examples['value'][i]}> "
            + f"{self.tokenizer.sep_token} "
            + examples["text"][i]
            for i in range(batch_size)
        ]
        return self.tokenizer(batched_inputs, truncation=True)

    def predict_values(self, author_dataset, prob_threshold=0.5):
        output = self.trainer.predict(author_dataset)
        y_pred = np.argmax(output.predictions, axis=1)
        return y_pred

    def create_user_dataset(self, user_comments, i):
        """
        Take in all texts from a single user, and output samples for each text and value combination
        """
        all_comments = [
            re.sub("http://\S+|https://\S+", "", x)
            for x in user_comments
            for _ in range(len(self.values))
        ]
        all_values = [x for _ in range(len(user_comments)) for x in self.values]
        user_dataset = Dataset.from_dict({"text": all_comments, "value": all_values})
        return user_dataset

    def extract_profile_from_predictions(
        self, user_comments, user_predictions, collapse_mft=None
    ):
        """
        Extract a profile from the sample predictions and create a data dump (for debugging).
        """
        profile = np.zeros((len(self.values),))
        prediction_dump = []
        for i, comment in enumerate(user_comments):
            active_values = []
            for j, value in enumerate(self.values):
                if user_predictions[(i * len(self.values)) + j] == 1:
                    profile[j] += 1
                    active_values.append(value)
            prediction_dump.append({"text": comment, "values": active_values})
        return profile, prediction_dump


class ValueEvalExtractor(ValueNetExtractor):
    def tokenize(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    def predict_values(self, author_dataset, prob_threshold=0.5):
        output = self.trainer.predict(author_dataset)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(output.predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= prob_threshold)] = 1
        return y_pred

    def create_user_dataset(self, user_comments, i):
        user_dataset = Dataset.from_dict({"text": user_comments})
        mask = np.ones((len(user_comments),)) * i
        return user_dataset, mask

    def extract_profile_from_predictions(
        self, user_dataset, user_predictions, collapse_mft=None
    ):
        text2preds = []
        for i in range(len(user_dataset)):
            row = user_dataset[i]
            prediction = user_predictions[i, :]
            text2preds.append({"text": row["text"], "prediction": prediction.tolist()})
        profile = user_predictions.sum(axis=0).reshape(-1)
        return profile, text2preds


class MFTValueExtractor(ValueExtractor):
    def __init__(
        self,
        load_path,
    ):
        super().__init__(load_path, "mft")
        self.label_mapping = self.trainer.model.config.id2label
        self.foundations = {
            i
            + 1: (
                (f_pos, self.trainer.model.config.label2id[f_pos]),
                (f_neg, self.trainer.model.config.label2id[f_neg]),
            )
            for i, (f_pos, f_neg) in enumerate(ValueConstants.MFT_FOUNDATIONS)
        }
        self.NON_MORAL_IDX = self.trainer.model.config.label2id["non-moral"]
        self.virtues = [
            self.trainer.model.config.label2id[x] for x in ValueConstants.MFT_VIRTUES
        ]
        self.vices = [
            self.trainer.model.config.label2id[x] for x in ValueConstants.MFT_VICES
        ]

    def tokenize(self, examples):
        batched_inputs = [m for m in examples["text"]]
        return self.tokenizer(batched_inputs, truncation=True)

    def predict_values(self, author_dataset, prob_threshold=0.5):
        output = self.trainer.predict(author_dataset)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(output.predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= prob_threshold)] = 1
        return y_pred

    def create_user_dataset(self, user_comments, i):
        user_dataset = Dataset.from_dict({"text": user_comments})
        mask = np.ones((len(user_comments),)) * i
        return user_dataset, mask

    def extract_profile_from_predictions(
        self, user_dataset, user_predictions, collapse_mft
    ):
        text2preds = []
        for i in range(len(user_dataset)):
            row = user_dataset[i]
            prediction = user_predictions[i, :]
            text2preds.append({"text": row["text"], "prediction": prediction.tolist()})
        profile = user_predictions.sum(axis=0).reshape(-1)
        if collapse_mft:
            profile = self.collapse_mft(profile)
        else:
            # Delete value entry for 'non-moral' prediction
            profile = np.delete(profile, self.NON_MORAL_IDX, 0)
        return profile, text2preds


class AutoValueExtractor:
    @staticmethod
    def create_extractor(load_path, override_type=None):
        if (
            "schwartz" in load_path
            or "valuenet" in load_path
            or override_type == "valuenet"
        ):
            return ValueNetExtractor(load_path)
        elif "mft" in load_path or override_type == "mft":
            return MFTValueExtractor(load_path)
        elif "valueeval" in load_path or override_type == "valueeval":
            return ValueEvalExtractor(load_path)
        else:
            raise ValueError(
                "Unable to interpret which model to load based on load_path."
            )


class ValueDictionary:
    def __init__(
        self,
        scoring_mechanism="any",
        aggregation_method=None,
        preprocessing=[],
        min_value_count=2,
    ):
        self.dictionary_filename = "data/Refined_dictionary.txt"
        self.dictionary_file_label_mapping = {
            1: "security",
            2: "conformity",
            3: "tradition",
            4: "benevolence",
            5: "universalism",
            6: "self-direction",
            7: "stimulation",
            8: "hedonism",
            9: "achievement",
            10: "power",
        }
        self.value_dictionary = defaultdict(lambda: [])
        self.label2id = {x: i for i, x in enumerate(ValueConstants.SCHWARTZ_VALUES)}
        self.id2label = {label_id: label for label, label_id in self.label2id.items()}
        self.scoring_mechanism = scoring_mechanism
        self.aggregation_method = aggregation_method
        self.preprocessing = preprocessing
        self.min_value_count = min_value_count

        with open(self.dictionary_filename, "r") as f:
            for line in f:
                word, value_id = line.split()
                self.value_dictionary[
                    self.dictionary_file_label_mapping[int(value_id)]
                ].append(word)
        self.num_dict_words = sum([len(x) for x in self.value_dictionary.values()])
        self.word_freq_weight = [
            len(self.value_dictionary[x]) / self.num_dict_words
            for x in ValueConstants.SCHWARTZ_VALUES
        ]

        if len(self.preprocessing) > 0:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def classify_comment_relevance(self, comment):
        label = "n"
        split_comment = comment.split()
        for value in self.value_dictionary:
            value_present = any(
                [(word in split_comment) for word in self.value_dictionary[value]]
            )
            if value_present:
                label = "y"
        return label

    def classify_comment_value(self, comment):
        # Optionally preprocess comment
        if "lemmatize" in self.preprocessing:
            doc = self.nlp(comment)
            " ".join([token.lemma_ for token in doc])

        # Split on spaces
        split_comment = comment.split()

        labels = []
        for value in self.value_dictionary:
            if self.scoring_mechanism == "any":
                value_present = any(
                    [(word in split_comment) for word in self.value_dictionary[value]]
                )
                if value_present:
                    labels.append(value)
            elif self.scoring_mechanism == "min_count":
                words_present = [
                    (word in split_comment) for word in self.value_dictionary[value]
                ]
                if sum(words_present) >= self.min_value_count:
                    labels.append(value)
            elif self.scoring_mechanism == "value_score":
                for word in self.value_dictionary[value]:
                    if word in split_comment:
                        labels.append(value)

        return labels

    def sum_normalize_profile(self, profile):
        """
        Normalize profile to sum to 1.

        if profile is zero, return profile
        """
        psum = np.sum(profile)
        if psum > 0:
            return profile / psum
        else:
            return profile

    def profile_multiple_users(self, comments_from_users):
        """
        Profile multiple users with a single run through the model.
        """
        user2profile = {}
        for user, uc in tqdm(
            comments_from_users.items(), total=len(comments_from_users)
        ):
            profile = [0] * len(ValueConstants.SCHWARTZ_VALUES)
            for comment in uc:
                labels = self.classify_comment_value(comment)
                for label in labels:
                    profile[self.label2id[label]] += 1
            user2profile[user] = np.array(profile)
            if self.aggregation_method == "sum_normalized":
                user2profile[user] = self.sum_normalize_profile(user2profile[user])
            elif self.aggregation_method == "freqweight_normalized":
                prof = user2profile[user] / self.word_freq_weight
                user2profile[user] = self.sum_normalize_profile(prof)
        return user2profile


class ValueConstants:
    MFT_VALUES = sorted(
        [
            "care",
            "fairness",
            "loyalty",
            "authority",
            "purity",
            "harm",
            "cheating",
            "betrayal",
            "subversion",
            "degradation",
            "non-moral",
        ]
    )
    MFT_FOUNDATIONS = [
        ("care", "harm"),
        ("fairness", "cheating"),
        ("loyalty", "betrayal"),
        ("authority", "subversion"),
        ("purity", "degradation"),
    ]
    MFT_VIRTUES = [
        "care",
        "fairness",
        "loyalty",
        "authority",
        "purity",
    ]
    MFT_VICES = [
        "harm",
        "cheating",
        "betrayal",
        "subversion",
        "degradation",
    ]
    SCHWARTZ_VALUES = sorted(
        [
            "achievement",
            "benevolence",
            "conformity",
            "hedonism",
            "power",
            "security",
            "self-direction",
            "stimulation",
            "tradition",
            "universalism",
        ]
    )

    SCHWARTZ_HIGHER_ORDER = sorted(
        {
            "conservation",
            "openness-to-change",
            "self-enhancement",
            "self-transcendence",
        }
    )
    SCHWARTZ_HIGHER_ORDER_MAPPING = {
        "achievement": "self-enhancement",
        "benevolence": "self-transcendence",
        "conformity": "conservation",
        "hedonism": "self-enhancement",
        "power": "self-enhancement",
        "self-direction": "openness-to-change",
        "security": "conservation",
        "stimulation": "openness-to-change",
        "tradition": "conservation",
        "universalism": "self-transcendence",
    }
    # Clockwise from the top: [Univ, Bene, Conf, Trad, Secu, Powe, Achi, Hedo, Stim, Self]
    SCHWARTZ_VALUES_CIRCUMPLEX_ORDER = [9, 1, 2, 8, 5, 4, 0, 3, 7, 6]

    # Normal distribution centered on value with sigma = 1
    SCHWARTZ_VALUE_SIMILARITY_MATRIX = np.array(
        [
            [
                1,
                0.3520653267642995,
                0.12951759566589174,
                0.01752830049356854,
                0.0008726826950457602,
                1.5983741106905478e-05,
                0.0008726826950457602,
                0.01752830049356854,
                0.12951759566589174,
                0.3520653267642995,
            ],
            [
                1.5983741106905478e-05,
                1,
                0.0008726826950457602,
                0.01752830049356854,
                0.12951759566589174,
                0.3520653267642995,
                0.3520653267642995,
                0.12951759566589174,
                0.01752830049356854,
                0.0008726826950457602,
            ],
            [
                0.12951759566589174,
                0.3520653267642995,
                1,
                0.3520653267642995,
                0.12951759566589174,
                0.01752830049356854,
                0.0008726826950457602,
                1.5983741106905478e-05,
                0.0008726826950457602,
                0.01752830049356854,
            ],
            [
                0.3520653267642995,
                0.12951759566589174,
                0.01752830049356854,
                1,
                0.0008726826950457602,
                1.5983741106905478e-05,
                0.0008726826950457602,
                0.01752830049356854,
                0.12951759566589174,
                0.3520653267642995,
            ],
            [
                0.12951759566589174,
                0.01752830049356854,
                0.0008726826950457602,
                1.5983741106905478e-05,
                1,
                0.0008726826950457602,
                0.01752830049356854,
                0.12951759566589174,
                0.3520653267642995,
                0.3520653267642995,
            ],
            [
                0.12951759566589174,
                0.01752830049356854,
                0.0008726826950457602,
                1.5983741106905478e-05,
                0.0008726826950457602,
                1,
                0.01752830049356854,
                0.12951759566589174,
                0.3520653267642995,
                0.3520653267642995,
            ],
            [
                0.3520653267642995,
                0.12951759566589174,
                0.01752830049356854,
                0.0008726826950457602,
                1.5983741106905478e-05,
                0.0008726826950457602,
                1,
                0.01752830049356854,
                0.12951759566589174,
                0.3520653267642995,
            ],
            [
                0.12951759566589174,
                0.3520653267642995,
                0.3520653267642995,
                0.12951759566589174,
                0.01752830049356854,
                0.0008726826950457602,
                1.5983741106905478e-05,
                1,
                0.0008726826950457602,
                0.01752830049356854,
            ],
            [
                1.5983741106905478e-05,
                0.0008726826950457602,
                0.01752830049356854,
                0.12951759566589174,
                0.3520653267642995,
                0.3520653267642995,
                0.12951759566589174,
                0.01752830049356854,
                1,
                0.0008726826950457602,
            ],
            [
                0.3520653267642995,
                0.12951759566589174,
                0.01752830049356854,
                0.0008726826950457602,
                1.5983741106905478e-05,
                0.0008726826950457602,
                0.01752830049356854,
                0.12951759566589174,
                0.3520653267642995,
                1,
            ],
        ]
    )

    # Normal distribution centered on value with sigma = 0.5
    SCHWARTZ_VALUE_SIMILARITY_MATRIX_05 = np.array(
        [
            [
                1,
                0.48394144903828673,
                0.008863696823876015,
                2.9734390294685958e-06,
                1.826944081672919e-11,
                2.0559547143337833e-18,
                1.826944081672919e-11,
                2.9734390294685958e-06,
                0.008863696823876015,
                0.48394144903828673,
            ],
            [
                2.0559547143337833e-18,
                1,
                1.826944081672919e-11,
                2.9734390294685958e-06,
                0.008863696823876015,
                0.48394144903828673,
                0.48394144903828673,
                0.008863696823876015,
                2.9734390294685958e-06,
                1.826944081672919e-11,
            ],
            [
                0.008863696823876015,
                0.48394144903828673,
                1,
                0.48394144903828673,
                0.008863696823876015,
                2.9734390294685958e-06,
                1.826944081672919e-11,
                2.0559547143337833e-18,
                1.826944081672919e-11,
                2.9734390294685958e-06,
            ],
            [
                0.48394144903828673,
                0.008863696823876015,
                2.9734390294685958e-06,
                1,
                1.826944081672919e-11,
                2.0559547143337833e-18,
                1.826944081672919e-11,
                2.9734390294685958e-06,
                0.008863696823876015,
                0.48394144903828673,
            ],
            [
                0.008863696823876015,
                2.9734390294685958e-06,
                1.826944081672919e-11,
                2.0559547143337833e-18,
                1,
                1.826944081672919e-11,
                2.9734390294685958e-06,
                0.008863696823876015,
                0.48394144903828673,
                0.48394144903828673,
            ],
            [
                0.008863696823876015,
                2.9734390294685958e-06,
                1.826944081672919e-11,
                2.0559547143337833e-18,
                1.826944081672919e-11,
                1,
                2.9734390294685958e-06,
                0.008863696823876015,
                0.48394144903828673,
                0.48394144903828673,
            ],
            [
                0.48394144903828673,
                0.008863696823876015,
                2.9734390294685958e-06,
                1.826944081672919e-11,
                2.0559547143337833e-18,
                1.826944081672919e-11,
                1,
                2.9734390294685958e-06,
                0.008863696823876015,
                0.48394144903828673,
            ],
            [
                0.008863696823876015,
                0.48394144903828673,
                0.48394144903828673,
                0.008863696823876015,
                2.9734390294685958e-06,
                1.826944081672919e-11,
                2.0559547143337833e-18,
                1,
                1.826944081672919e-11,
                2.9734390294685958e-06,
            ],
            [
                2.0559547143337833e-18,
                1.826944081672919e-11,
                2.9734390294685958e-06,
                0.008863696823876015,
                0.48394144903828673,
                0.48394144903828673,
                0.008863696823876015,
                2.9734390294685958e-06,
                1,
                1.826944081672919e-11,
            ],
            [
                0.48394144903828673,
                0.008863696823876015,
                2.9734390294685958e-06,
                1.826944081672919e-11,
                2.0559547143337833e-18,
                1.826944081672919e-11,
                2.9734390294685958e-06,
                0.008863696823876015,
                0.48394144903828673,
                1,
            ],
        ]
    )

    # Normal distribution centered on value with sigma = 2
    SCHWARTZ_VALUE_SIMILARITY_MATRIX_2 = np.array(
        [
            [
                1,
                0.19333405840142465,
                0.15056871607740221,
                0.09132454269451096,
                0.043138659413255766,
                0.01586982591783371,
                0.043138659413255766,
                0.09132454269451096,
                0.15056871607740221,
                0.19333405840142465,
            ],
            [
                0.01586982591783371,
                1,
                0.043138659413255766,
                0.09132454269451096,
                0.15056871607740221,
                0.19333405840142465,
                0.19333405840142465,
                0.15056871607740221,
                0.09132454269451096,
                0.043138659413255766,
            ],
            [
                0.15056871607740221,
                0.19333405840142465,
                1,
                0.19333405840142465,
                0.15056871607740221,
                0.09132454269451096,
                0.043138659413255766,
                0.01586982591783371,
                0.043138659413255766,
                0.09132454269451096,
            ],
            [
                0.19333405840142465,
                0.15056871607740221,
                0.09132454269451096,
                1,
                0.043138659413255766,
                0.01586982591783371,
                0.043138659413255766,
                0.09132454269451096,
                0.15056871607740221,
                0.19333405840142465,
            ],
            [
                0.15056871607740221,
                0.09132454269451096,
                0.043138659413255766,
                0.01586982591783371,
                1,
                0.043138659413255766,
                0.09132454269451096,
                0.15056871607740221,
                0.19333405840142465,
                0.19333405840142465,
            ],
            [
                0.15056871607740221,
                0.09132454269451096,
                0.043138659413255766,
                0.01586982591783371,
                0.043138659413255766,
                1,
                0.09132454269451096,
                0.15056871607740221,
                0.19333405840142465,
                0.19333405840142465,
            ],
            [
                0.19333405840142465,
                0.15056871607740221,
                0.09132454269451096,
                0.043138659413255766,
                0.01586982591783371,
                0.043138659413255766,
                1,
                0.09132454269451096,
                0.15056871607740221,
                0.19333405840142465,
            ],
            [
                0.15056871607740221,
                0.19333405840142465,
                0.19333405840142465,
                0.15056871607740221,
                0.09132454269451096,
                0.043138659413255766,
                0.01586982591783371,
                1,
                0.043138659413255766,
                0.09132454269451096,
            ],
            [
                0.01586982591783371,
                0.043138659413255766,
                0.09132454269451096,
                0.15056871607740221,
                0.19333405840142465,
                0.19333405840142465,
                0.15056871607740221,
                0.09132454269451096,
                1,
                0.043138659413255766,
            ],
            [
                0.19333405840142465,
                0.15056871607740221,
                0.09132454269451096,
                0.043138659413255766,
                0.01586982591783371,
                0.043138659413255766,
                0.09132454269451096,
                0.15056871607740221,
                0.19333405840142465,
                1,
            ],
        ]
    )

    # Normal distribution centered on value with sigma = 5
    SCHWARTZ_VALUE_SIMILARITY_MATRIX_5 = np.array(
        [
            [
                1,
                0.07939050949540236,
                0.07627756309210483,
                0.0704130653528599,
                0.06245078667335226,
                0.05321704997975096,
                0.06245078667335226,
                0.0704130653528599,
                0.07627756309210483,
                0.07939050949540236,
            ],
            [
                0.05321704997975096,
                1,
                0.06245078667335226,
                0.0704130653528599,
                0.07627756309210483,
                0.07939050949540236,
                0.07939050949540236,
                0.07627756309210483,
                0.0704130653528599,
                0.06245078667335226,
            ],
            [
                0.07627756309210483,
                0.07939050949540236,
                1,
                0.07939050949540236,
                0.07627756309210483,
                0.0704130653528599,
                0.06245078667335226,
                0.05321704997975096,
                0.06245078667335226,
                0.0704130653528599,
            ],
            [
                0.07939050949540236,
                0.07627756309210483,
                0.0704130653528599,
                1,
                0.06245078667335226,
                0.05321704997975096,
                0.06245078667335226,
                0.0704130653528599,
                0.07627756309210483,
                0.07939050949540236,
            ],
            [
                0.07627756309210483,
                0.0704130653528599,
                0.06245078667335226,
                0.05321704997975096,
                1,
                0.06245078667335226,
                0.0704130653528599,
                0.07627756309210483,
                0.07939050949540236,
                0.07939050949540236,
            ],
            [
                0.07627756309210483,
                0.0704130653528599,
                0.06245078667335226,
                0.05321704997975096,
                0.06245078667335226,
                1,
                0.0704130653528599,
                0.07627756309210483,
                0.07939050949540236,
                0.07939050949540236,
            ],
            [
                0.07939050949540236,
                0.07627756309210483,
                0.0704130653528599,
                0.06245078667335226,
                0.05321704997975096,
                0.06245078667335226,
                1,
                0.0704130653528599,
                0.07627756309210483,
                0.07939050949540236,
            ],
            [
                0.07627756309210483,
                0.07939050949540236,
                0.07939050949540236,
                0.07627756309210483,
                0.0704130653528599,
                0.06245078667335226,
                0.05321704997975096,
                1,
                0.06245078667335226,
                0.0704130653528599,
            ],
            [
                0.05321704997975096,
                0.06245078667335226,
                0.0704130653528599,
                0.07627756309210483,
                0.07939050949540236,
                0.07939050949540236,
                0.07627756309210483,
                0.0704130653528599,
                1,
                0.06245078667335226,
            ],
            [
                0.07939050949540236,
                0.07627756309210483,
                0.0704130653528599,
                0.06245078667335226,
                0.05321704997975096,
                0.06245078667335226,
                0.0704130653528599,
                0.07627756309210483,
                0.07939050949540236,
                1,
            ],
        ]
    )

    # Normal distribution centered on value with sigma = 0.2
    SCHWARTZ_VALUE_SIMILARITY_MATRIX_02 = np.array(
        [
            [
                1,
                0.08764150246784269,
                1.217160266514505e-12,
                2.347597678987573e-34,
                6.288361914390968e-67,
                2.3393184086250235e-110,
                6.288361914390968e-67,
                2.347597678987573e-34,
                1.217160266514505e-12,
                0.08764150246784269,
            ],
            [
                2.3393184086250235e-110,
                1,
                6.288361914390968e-67,
                2.347597678987573e-34,
                1.217160266514505e-12,
                0.08764150246784269,
                0.08764150246784269,
                1.217160266514505e-12,
                2.347597678987573e-34,
                6.288361914390968e-67,
            ],
            [
                1.217160266514505e-12,
                0.08764150246784269,
                1,
                0.08764150246784269,
                1.217160266514505e-12,
                2.347597678987573e-34,
                6.288361914390968e-67,
                2.3393184086250235e-110,
                6.288361914390968e-67,
                2.347597678987573e-34,
            ],
            [
                0.08764150246784269,
                1.217160266514505e-12,
                2.347597678987573e-34,
                1,
                6.288361914390968e-67,
                2.3393184086250235e-110,
                6.288361914390968e-67,
                2.347597678987573e-34,
                1.217160266514505e-12,
                0.08764150246784269,
            ],
            [
                1.217160266514505e-12,
                2.347597678987573e-34,
                6.288361914390968e-67,
                2.3393184086250235e-110,
                1,
                6.288361914390968e-67,
                2.347597678987573e-34,
                1.217160266514505e-12,
                0.08764150246784269,
                0.08764150246784269,
            ],
            [
                1.217160266514505e-12,
                2.347597678987573e-34,
                6.288361914390968e-67,
                2.3393184086250235e-110,
                6.288361914390968e-67,
                1,
                2.347597678987573e-34,
                1.217160266514505e-12,
                0.08764150246784269,
                0.08764150246784269,
            ],
            [
                0.08764150246784269,
                1.217160266514505e-12,
                2.347597678987573e-34,
                6.288361914390968e-67,
                2.3393184086250235e-110,
                6.288361914390968e-67,
                1,
                2.347597678987573e-34,
                1.217160266514505e-12,
                0.08764150246784269,
            ],
            [
                1.217160266514505e-12,
                0.08764150246784269,
                0.08764150246784269,
                1.217160266514505e-12,
                2.347597678987573e-34,
                6.288361914390968e-67,
                2.3393184086250235e-110,
                1,
                6.288361914390968e-67,
                2.347597678987573e-34,
            ],
            [
                2.3393184086250235e-110,
                6.288361914390968e-67,
                2.347597678987573e-34,
                1.217160266514505e-12,
                0.08764150246784269,
                0.08764150246784269,
                1.217160266514505e-12,
                2.347597678987573e-34,
                1,
                6.288361914390968e-67,
            ],
            [
                0.08764150246784269,
                1.217160266514505e-12,
                2.347597678987573e-34,
                6.288361914390968e-67,
                2.3393184086250235e-110,
                6.288361914390968e-67,
                2.347597678987573e-34,
                1.217160266514505e-12,
                0.08764150246784269,
                1,
            ],
        ]
    )
