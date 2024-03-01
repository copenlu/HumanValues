import argparse
import pickle
from collections import defaultdict
from typing import List, Dict

import pandas as pd
import spacy
import torch
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer
from value_disagreement.extraction import ValueNetExtractor

# Load English language model
nlp = spacy.load("en_core_web_sm")


tokenizer_override = "microsoft/deberta-v3-base"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Label subreddits")
    parser.add_argument("posts_path", type=str, help="Path to the subreddit posts")
    parser.add_argument(
        "model_checkpoint", type=str, help="Path to the model checkpoint"
    )
    parser.add_argument("save_dir", type=str, help="Path to save the data")
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--total", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def filter_non_english(
    comments: List[str], detector: LanguageDetectorBuilder
) -> List[str]:
    """
    Filter out non-English comments.
    """
    return [c for c in comments if detector.detect_language_of(c) == Language.ENGLISH]


def get_subreddit_posts_random_sample(
    subreddit: List[str], posts=True, is_tuple=True
) -> List[str]:
    """
    Get a random sample of subreddit posts.
    """
    if posts:
        texts = (
            [post[1] + post[2] for post in subreddit]
            if is_tuple
            else [post["selftext"] + post["title"] for post in subreddit]
        )
    else:
        texts = (
            [comment[1] for comment in subreddit]
            if is_tuple
            else [comment["body"] for comment in subreddit]
        )

    return [c for c in texts if c is not None and len(c) > 0]


def probs_to_dict_of_probs(
    ve: ValueNetExtractor, comments: List[str], probs: torch.Tensor
) -> Dict[str, float]:
    """
    Convert probabilities to a dictionary of probabilities.
    """
    probs_dict = defaultdict(float)
    for i in range(len(comments)):
        comment_probs = probs[i * len(ve.values) : (i + 1) * len(ve.values)]
        for prob, value in zip(comment_probs, ve.values):
            probs_dict[value] += prob
    probs_dict = {
        k: v / len(comments) for k, v in probs_dict.items()
    }  # average over comments
    return {k: v.item() for k, v in probs_dict.items()}


def get_probabilities(ve: ValueNetExtractor, comments: List[str]) -> torch.Tensor:
    """
    Get probabilities for each comment.
    """
    user_dataset = ve.create_user_dataset(comments, 0)
    user_dataset = user_dataset.map(ve.tokenize, batched=True)
    predictions = ve.trainer.predict(user_dataset).predictions
    probabilities = torch.softmax(torch.tensor(predictions), dim=1)
    return probabilities[:, 1]


def main(args):
    ve = ValueNetExtractor(args.model_checkpoint, batch_size=args.batch_size)
    ve.tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_override
    )  
    detector = LanguageDetectorBuilder.from_all_languages().build()
    print("loaded model")

    data = pickle.load(open(args.posts_path, "rb"))
    metadata = {k: {} for k in data.keys()}
    print("loaded data")

    failed = set()
    subreddit_values = dict()
    subreddits = list(data.keys())
    subreddits.sort()
    subreddits = subreddits[args.subset :: args.total]

    save_path = args.save_dir if args.total == 1 else f"{args.save_dir}_{args.subset}"
    
    for i, subreddit in enumerate(subreddits):
        try:
            posts = get_subreddit_posts_random_sample(data[subreddit])
        except:
            continue
        english_posts = filter_non_english(posts, detector)
        if len(english_posts) < 250:
            continue
        print(subreddit)
        print(len(english_posts))
        try:
            probs = get_probabilities(ve, english_posts)
        except RuntimeError:
            print("RuntimeError: ", subreddit)
            failed.add(subreddit)
            continue
        probs_dict = probs_to_dict_of_probs(ve, english_posts, probs)
        probs_dict.update(metadata[subreddit])
        probs_dict["num_total_posts"] = len(posts)
        probs_dict["num_english_posts"] = len(english_posts)
        subreddit_values[subreddit] = probs_dict
        print(probs_dict)
        if i % 100 == 0:
            df = pd.DataFrame(subreddit_values).T
            df.to_csv(f"{save_path}/subreddit_schwarts_values.csv")

    df = pd.DataFrame(subreddit_values).T
    df.to_csv(f"{save_path}/subreddit_schwarts_values.csv")
    print(failed)
    pickle.dump(failed, open(f"{save_path}/subreddit_schwarts_value_failed.pkl", "wb"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
