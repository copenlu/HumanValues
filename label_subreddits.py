import pickle
import torch
from value_disagreement.extraction import ValueNetExtractor
from transformers import AutoTokenizer
from collections import defaultdict
import pandas as pd
from lingua import Language, LanguageDetectorBuilder
import argparse
import spacy

nlp = spacy.load("en_core_web_sm")

ROOT_DIR = "[ROOT_DIR]"
model_checkpoint = f"{ROOT_DIR}/value_extractor_model"
tokenizer_override = "microsoft/deberta-v3-base"
SUBREDDITS_PATH = "[SUBREDDITS_PATH]"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, required=False, default=0)
    parser.add_argument("--total", type=int, required=False, default=1)
    parser.add_argument("--batch_size", type=int, required=False, default=8)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--broken_subs", action="store_true")
    return parser.parse_args()


def filter_non_english(comments, detector):
    filtered_comments = [
        c for c in comments if detector.detect_language_of(c) == Language.ENGLISH
    ]
    return filtered_comments


def get_subreddit_posts_random_sample(subreddit, posts=True, is_tuple=True):
    if posts:
        if not is_tuple:
            texts = [post["selftext"] + post["title"] for post in subreddit]
        else:
            texts = [post[1] + post[2] for post in subreddit]
    else:
        if not is_tuple:
            texts = [comment["body"] for comment in subreddit]
        else:
            texts = [comment[1] for comment in subreddit]
    texts = [c for c in texts if c is not None and len(c) > 0]
    return texts


def probs_to_dict_of_probs(ve, comments, probs):
    probs_dict = defaultdict(float)
    for i in range(len(comments)):
        comment_probs = probs[i * len(ve.values) : (i + 1) * len(ve.values)]
        for prob, value in zip(comment_probs, ve.values):
            probs_dict[value] += prob
    probs_dict = {
        k: v / len(comments) for k, v in probs_dict.items()
    }  # average over comments
    probs_dict = {k: v.item() for k, v in probs_dict.items()}
    return probs_dict


def get_probabilities(ve, comments):
    user_dataset = ve.create_user_dataset(comments, 0)
    user_dataset = user_dataset.map(ve.tokenize, batched=True)
    predictions = ve.trainer.predict(user_dataset).predictions
    probabilities = torch.softmax(torch.tensor(predictions), dim=1)
    probabilities = probabilities[:, 1]
    return probabilities


def main(args):
    ve = ValueNetExtractor(model_checkpoint, batch_size=args.batch_size)
    ve.tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_override
    )  # we need to override the tokenizer because their code is weird
    detector = LanguageDetectorBuilder.from_all_languages().build()
    print("loaded model")

    data = pickle.load(open(SUBREDDITS_PATH, "rb"))
    metadata = {k: {} for k in data.keys()}
    print("loaded data")

    failed = set()
    subreddit_values = dict()
    subreddits = list(data.keys())
    subreddits.sort()
    subreddits = subreddits[args.subset :: args.total]

    for i, subreddit in enumerate(subreddits):
        if len(data[subreddit]) < 100:
            continue
        try:
            comments = get_subreddit_posts_random_sample(data[subreddit])
        except:
            continue
        english_comments = filter_non_english(comments, detector)
        if len(english_comments) < 250:
            continue
        print(subreddit)
        print(len(english_comments))
        try:
            probs = get_probabilities(ve, english_comments)
        except RuntimeError:
            print("RuntimeError: ", subreddit)
            failed.add(subreddit)
            continue
        probs_dict = probs_to_dict_of_probs(ve, english_comments, probs)
        probs_dict.update(metadata[subreddit])
        probs_dict["num_total_comments"] = len(comments)
        probs_dict["num_english_comments"] = len(english_comments)
        subreddit_values[subreddit] = probs_dict
        print(probs_dict)
        if i % 100 == 0:
            df = pd.DataFrame(subreddit_values).T
            df.to_csv(f"{args.save_path}_{args.subset}.csv")

    df = pd.DataFrame(subreddit_values).T
    df.to_csv(f"{args.save_path}_{args.subset}.csv")
    print(failed)
    pickle.dump(failed, open(f"{args.save_path}_{args.subset}_failed.pkl", "wb"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
