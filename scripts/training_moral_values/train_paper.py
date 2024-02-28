import wandb
import argparse
import json
import sys
import os

sys.path.append(".")
from datasets import concatenate_datasets
from ray import tune
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from value_disagreement.datasets import (
    RedditAnnotatedDataset,
    ValueEvalDataset,
    ValueNetDataset
)
from value_disagreement.datasets.utils import cast_dataset_to_hf, hf_dataset_tokenize
from value_disagreement.evaluation import single_label_metrics_cls
from value_disagreement.extraction import ValueConstants, ValueTokenizer
from value_disagreement.utils import print_stats, seed_everything


def get_dataset(args):
    """
    Load the appropriate dataset based on the input arguments.
    """
    if args.train_dataset == "both":
        # Combined dataset
        dataset_net = ValueNetDataset(
            f"{args.project_dir}/data/valuenet/", return_predefined_splits=True
        )
        dataset_eval = ValueEvalDataset(
            f"{args.project_dir}/data/valueeval/",
            cast_to_valuenet=args.no_label_remap,
            return_predefined_splits=True,
        )
        train_vn_idx, val_vn_idx, test_vn_idx = dataset_net.get_splits()
        train_ve_idx, val_ve_idx, test_ve_idx = dataset_eval.get_splits()
        vn_train_hf = cast_dataset_to_hf(dataset_net[train_vn_idx], "train")
        vn_val_hf = cast_dataset_to_hf(dataset_net[val_vn_idx], "val")
        vn_test_hf = cast_dataset_to_hf(dataset_net[test_vn_idx], "test")
        ve_train_hf = cast_dataset_to_hf(dataset_eval[train_ve_idx], "train")
        ve_val_hf = cast_dataset_to_hf(dataset_eval[val_ve_idx], "val")
        ve_test_hf = cast_dataset_to_hf(dataset_eval[test_ve_idx], "test")
        train_hf = concatenate_datasets([vn_train_hf, ve_train_hf])
        val_hf = concatenate_datasets([vn_val_hf, ve_val_hf])
        test_hf = concatenate_datasets([vn_test_hf, ve_test_hf])
    else:
        # Single dataset
        if args.train_dataset == "valuenet":
            dataset = ValueNetDataset(
                f"{args.project_dir}/data/valuenet/",
                return_predefined_splits=True,
            )
        elif args.train_dataset == "valueeval":
            dataset = ValueEvalDataset(
                f"{args.project_dir}/data/valueeval/",
                cast_to_valuenet=args.no_label_remap,
                return_predefined_splits=True,
            )
        elif args.train_dataset == "climatesingle-small":
            dataset = RedditAnnotatedDataset(
                "data/annotated_data/climatesingle_small.csv"
            )
        elif args.train_dataset == "climatesingle-large":
            dataset = RedditAnnotatedDataset(
                "data/annotated_data/climatesingle_large.csv"
            )
        train_idx, val_idx, test_idx = dataset.get_splits()
        train_hf = cast_dataset_to_hf(dataset[train_idx], "train")
        val_hf = cast_dataset_to_hf(dataset[val_idx], "val")
        test_hf = cast_dataset_to_hf(dataset[test_idx], "test")
    return train_hf, val_hf, test_hf


def model_init(checkpoint, tokenizer):
    """
    Initialize the model with the checkpoint and tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, problem_type="single_label_classification"
    )
    model.resize_token_embeddings(len(tokenizer.tokenizer))
    return model


def dump_predictions(args, dataset, results, output_file):
    """
    Dump model predictions to a file.
    """
    os.makedirs("output/predictions", exist_ok=True)
    os.makedirs(output_file[: output_file.rfind("/")], exist_ok=True)
    with open(output_file, "w") as f:
        result_dump = {
            "args": vars(args),
            "preds": results.predictions.tolist(),
            "true": dataset["labels"].tolist(),
            "metrics": results.metrics,
        }
        json.dump(result_dump, f)


def main(args):
    seed_everything(args.seed)
    train_hf, val_hf, test_hf = get_dataset(args)
    print(
        f"Sizes of sets: train: {len(train_hf)}, val: {len(val_hf)}, test: {len(test_hf)}"
    )

    # Load the right checkpoint / model definition for tokenizer
    if args.checkpoint is None:
        checkpoint = args.use_model
    else:
        checkpoint = args.checkpoint

    # Construct tokenizer
    value_tokenizer = ValueTokenizer(
        args.use_model, input_concat=True, label_type="cast_float"
    )

    # Tokenize data
    tokenized_dataset_train = hf_dataset_tokenize(
        train_hf, value_tokenizer, args.use_model
    )
    tokenized_dataset_val = hf_dataset_tokenize(val_hf, value_tokenizer, args.use_model)
    tokenized_dataset_test = hf_dataset_tokenize(
        test_hf, value_tokenizer, args.use_model
    )

    # Add special value tokens and initalize model
    num_added_tokens = value_tokenizer.tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                f"<{x}>" for x in ValueConstants.SCHWARTZ_VALUES
            ]
        }
    )
    print(f"Added {num_added_tokens} special value tokens")
    model = model_init(checkpoint, value_tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        f"value_estimation_trainer_{args.train_dataset}_new_tok",
        report_to="none" if args.eval_only else "wandb",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        metric_for_best_model="eval_f1",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
    )

    # Compute metrics function
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        return single_label_metrics_cls(preds, p.label_ids)

    # Setup training
    data_collator = DataCollatorWithPadding(tokenizer=value_tokenizer.tokenizer)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_val,
        data_collator=data_collator,
        tokenizer=value_tokenizer.tokenizer,
        compute_metrics=compute_metrics,
    )

    if args.predict_only:
        results = trainer.predict(tokenized_dataset_test)
        dump_predictions(
            tokenized_dataset_test,
            results,
            f"output/predictions/moral_values_{args.train_dataset}.json",
        )
    elif args.eval_only:
        print(trainer.evaluate(tokenized_dataset_test))
    else:
        # Run training
        if args.ray:

            def hyper_space(x):
                return {
                    "learning_rate": tune.loguniform(1e-7, 1e-3),
                    "num_train_epochs": tune.choice(list(range(1, 10))),
                    "seed": tune.uniform(1, 42),
                    "weight_decay": tune.uniform(1e-7, 1e-4),
                    "lr_scheduler_type": tune.choice(["cosine", "linear"]),
                }

            trainer = Trainer(
                args=training_args,
                model_init=lambda: model_init(checkpoint, value_tokenizer),
                train_dataset=tokenized_dataset_train,
                eval_dataset=tokenized_dataset_val,
                data_collator=data_collator,
                tokenizer=value_tokenizer.tokenizer,
                compute_metrics=compute_metrics,
            )
            trainer.hyperparameter_search(
                hp_space=hyper_space,
                direction="maximize",
                backend="ray",
                n_trials=20,
                local_dir=f"{args.project_dir}/ray_results_moral_values/",
            )
        else:
            reses = []
            for i in range(args.n_runs):
                seed_everything(i)
                model = model_init(checkpoint, value_tokenizer)
                trainer = Trainer(
                    model,
                    training_args,
                    train_dataset=tokenized_dataset_train,
                    eval_dataset=tokenized_dataset_val,
                    data_collator=data_collator,
                    tokenizer=value_tokenizer.tokenizer,
                    compute_metrics=compute_metrics,
                )
                trainer.train()
                model_name = f"moral_values_trainer_{args.use_model}_{args.train_dataset}_{i}_new_tok"
                results = trainer.predict(tokenized_dataset_test)
                dump_predictions(
                    args,
                    tokenized_dataset_test,
                    results,
                    f"output/predictions/{model_name}.json",
                )
                trainer.save_model(model_name)
                reses.append({"model_name": model_name, "results": results})
            print_stats(
                "Precision", [res["results"].metrics["test_precision"] for res in reses]
            )
            print_stats(
                "Recall", [res["results"].metrics["test_recall"] for res in reses]
            )
            print_stats("F1", [res["results"].metrics["test_f1"] for res in reses])
            print_stats(
                "Accuracy", [res["results"].metrics["test_accuracy"] for res in reses]
            )


if __name__ == "__main__":
    wandb.init(project="Creddit")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_dataset",
        choices=[
            "valuenet",
            "valueeval",
            "climatesingle-small",
            "climatesingle-large",
            "both",
        ],
        help="which dataset to train on (infers value type)",
    )
    parser.add_argument(
        "--project_dir", type=str, default=".", help="Path to the project folder"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-05,
        help="Learning rate at the start of training",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=0, help="Reproducibility seed"
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, help="Number of times to train a model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Number of samples in a batch"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay hyperparameter"
    )
    parser.add_argument(
        "--use_model", type=str, default=None, help="Model config to load"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to load"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Only do evaluation on test set",
    )
    parser.add_argument(
        "--no_label_remap",
        action="store_false",
        default=True,
        help="Do not remap labels in the dataset to a common set",
    )
    parser.add_argument(
        "--predict_only",
        action="store_true",
        default=False,
        help="Only do prediction on test set",
    )
    parser.add_argument(
        "--ray",
        default=False,
        action="store_true",
        help="If true, use ray for distributed hyperparameter search.",
    )
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
