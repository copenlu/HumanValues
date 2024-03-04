import json
import torch
import random
import signal
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from value_disagreement.extraction import ValueConstants
from value_disagreement.extraction import ValueNetExtractor
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT_DIR = "../"

random.seed(12)

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def get_list_of_subreddits()-> list:
    subreddit_value_df = pd.read_csv(f"{ROOT_DIR}/data/normalised_subreddit_values.csv")
    subreddits = subreddit_value_df["subreddit"].tolist()
    return subreddits

def count_word_occurrences(list_of_words:list, target_string:list)-> int:
    count = 0
    for word in list_of_words:
        count += target_string.count(word)
    return count

def convert_value_dict(path_to_value_dict:str="../data/personal-values-dictionary.dicx"):
    """Converts the Schwartz Value Dictionary from .dicx to .csv format."""

    value_df = pd.read_csv(path_to_value_dict)
    value_df = value_df.fillna("0")
    value_df = value_df.replace("X",1)
    value_df = value_df.set_index("DicTerm")
    value_df.to_csv(f"{ROOT_DIR}/data/value_dict.csv")
    return 

def generate_synthetic_posts(total_generations:int)->None:
    """Generate synthetic value-oriented posts for using Mistral-7B-Instruct-v0.2 model"""
    
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
        load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side="left")
    
    tokenizer.pad_token = tokenizer.eos_token

    # Can be replaced with any list of subreddits
    list_of_subreddits = get_list_of_subreddits()
    random.shuffle(list_of_subreddits)

    # Get a list of words for each value from the lexica
    value_df = pd.read_csv(f"{ROOT_DIR}/data/value_dict.csv", index_col=[0])
    selected_values = [val.replace("-","") for val in ValueConstants.SCHWARTZ_VALUES]
    value_df.columns = value_df.columns.str.lower()
    value_df = value_df[selected_values]
    
    num_generations = 0

    for subreddit in list_of_subreddits:
        if num_generations > total_generations:
            break
        for value, value_word_indices in value_df.items():
            if num_generations > total_generations:
                break
            
            # Get list of words for the value
            value_word_list = value_word_indices[value_word_indices == 1].index.tolist()
            
            # Generate 5 posts for each subreddit, value pair
            for i in range(5):
                if num_generations > total_generations:
                    break

                # Select 10 words from the value word list
                value_words = random.sample(value_word_list, 10)
                prompt = "Generate text that resembles a post on Reddit from the subreddit: '{subreddit}'.\
                        In the post, add words from the this list of words: {value_words}\
                        Make sure that the text of the post includes some of the above words, is fluent, and is realistic for the subreddit.\
                        Provide the output in a json format with a 'title' for the post and the 'text' as fields."

                model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
                try:
                    with timeout(seconds=30):
                        generated_ids = model.generate(**model_inputs, max_new_tokens=5000, pad_token_id=tokenizer.pad_token_id)
                        generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    # Extract the json from the generated string
                    generated_string = generated_string.split("```json")[1].split("```")[0]
                    post_json = json.loads(generated_string)
                    post_json["text"] = post_json["text"].replace("\n", " ")
                    post_json["subreddit"] = subreddit
                    post_json["value"] = value
                    post_json["id"] = num_generations
                    post_json["value_words"] = value_words
                    post_json["value_word_count"] = count_word_occurrences(value_words, post_json["text"])

                    with open(f"{ROOT_DIR}/data/synthetic_value_posts.jsonl", "a") as f:
                        json.dump(post_json, f)
                        f.write("\n")
                    num_generations += 1
                except TimeoutError:
                    print("Timeout")
                    continue
                except Exception as e:
                    print(e)
                    continue
    return  f"{ROOT_DIR}/data/synthetic_value_posts.jsonl"              

def get_classifier_probabilities(ve, comments):
    user_dataset = ve.create_user_dataset(comments, 0)
    user_dataset = user_dataset.map(ve.tokenize, batched=True)
    predictions = ve.trainer.predict(user_dataset).predictions
    probabilities = torch.softmax(torch.tensor(predictions), dim=1)
    probabilities = probabilities[:, 1]
    return probabilities

def evaluate_classifier(model_checkpoint:str="microsoft/deberta-v3-base", synthetic_posts_path:str="synthetic_value_posts.jsonl"):
    """Evaluate the classifier on the synthetic posts"""

    # Load the model
    ve = ValueNetExtractor(model_checkpoint)
    ve.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)  # we need to override the tokenizer because their code is weird

    # Load the data
    try:
        comments_df = pd.read_json(f"{ROOT_DIR}/data/synthetic_value_posts.jsonl", lines=True)
    except FileNotFoundError:
        print("Synthetic posts missing or in the wrong format. Please generate synthetic posts first.")
        return
    comments_df.set_index("id", inplace=True)
    comments = comments_df['text'].values.tolist()
    labels = comments_df['value'].values.tolist()
    label_names = list(ve.label_mapping.values())
    label_names = [label.replace("-", "") for label in label_names]

    probs = get_classifier_probabilities(ve, comments)
    
    predictions = []
    prediction_probs = []
    for i in range(0,len(probs),10):
        ps = probs[i:i+10] # get_classifier_probabilities() returns concatenated probs
        pred_index = torch.argmax(ps)
        pred_prob = ps[pred_index]
        pred_label = label_names[pred_index]
        prediction_probs.append(pred_prob)
        predictions.append(pred_label)

    ranks = []
    for i, lab in zip(range(0,len(probs),10), labels):
        ps = probs[i:i+10]
        desc_indices = torch.argsort(ps, descending=True)
        targ_label_index = label_names.index(lab)
        ranks.append(desc_indices.tolist().index(targ_label_index))

    comments_df["predictions"] = predictions
    comments_df["prediction_probs"] = prediction_probs
    comments_df.to_csv(f"{ROOT_DIR}/data/synthetic_post_predictions.csv")

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    
    results = {"predictions": predictions, "labels": labels, "ranks": ranks, 
               "accuracy": accuracy, "f1": f1}
    
    with open(f"{ROOT_DIR}/results/value_classifier_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Value model checkpoint to use for evaluation")
    parser.add_argument(
        "--do_generation",
        type=bool,
        default=True,
        help="Generate synthetic posts")
    parser.add_argument(
        "--total_generations",
        type=int,
        default=6000,
        help="Total number of synthetic posts to generate")
    parser.add_argument(
        "--synthetic_posts_path",
        type=str,
        default="synthetic_value_posts.jsonl",
        help="Path to the synthetic posts file"
    )
    args = parser.parse_args()

    convert_value_dict()
    if args.do_generation:
        posts_path = generate_synthetic_posts(args.total_generations, args.model_checkpoint)
        args.synthetic_posts_path = posts_path
    evaluate_classifier(args.model_checkpoint, args.synthetic_posts_path)
