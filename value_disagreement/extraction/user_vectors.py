import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from value_disagreement.extraction import (ValueConstants, ValueDictionary,
                                           ValueNetExtractor)


def dump_vectors(vectors, outfile):
    """
    Dump user vectors to JSON file.
    """
    with open(outfile, "w") as f:
        json.dump(vectors, f)


def get_user_vectors(pipeline, user2data):
    """
    Run the given sklearn pipeline over the user data to obtain average user vectors.
    """
    stemmer = SnowballStemmer(language='english')
    user2vector = {}
    for user in tqdm(user2data):
        v = pipeline.transform([stemmer.stem(x) for x in user2data[user]])
        c = v.mean(axis=0)
        c_squeezed = np.asarray(c).squeeze()
        user2vector[user] = c_squeezed.tolist()
    return user2vector


def get_user2data(all_subreddits_posts):
    """
    Combine all data from a single users across all subreddits.
    """
    user2data = defaultdict(lambda: [])
    for sr in all_subreddits_posts:
        for user, data in sr.user2data.items():
            user2data[user].extend(data)
    return user2data


def get_centroids(all_subreddits_posts, debagreement_text, max_features=768, prefilter=False):
    """
    Get TF-IDF centroids for all users. Generate TF-IDF vocabulary from the Debagreement data.

    If prefilter is True, use only the value-laden comments from debagreement.

    """
    stemmer = SnowballStemmer(language='english')
    corpus = []
    filename_suffix = f"_{max_features}"
    if prefilter:
        vd = ValueDictionary(
            scoring_mechanism='any',
            aggregation_method=None,
            preprocessing='lemmatize'
        )
        value_comments = []
        for text in debagreement_text:
            if vd.classify_comment_relevance(text) == "y":
                value_comments.append(text)
        debagreement_text = value_comments
        filename_suffix = "_prefiltered"

    for text in debagreement_text:
        corpus.append(stemmer.stem(text))

    pp = Pipeline([
        ('vect', CountVectorizer(max_features=max_features)),
        ('tfidf', TfidfTransformer()),
     ])
    pp.fit(corpus)

    user2data = get_user2data(all_subreddits_posts)
    user2vector = get_user_vectors(pp, user2data)
    dump_vectors(user2vector, f"data/user_centroids{filename_suffix}.json")


def get_values(
        all_subreddits_posts,
        preprocessing=[],
        aggregation_method=None,
        value_estimation="dictionary",
        checkpoint=None):
    """
    Get value profiles for all users
    """
    if value_estimation == "dictionary":
        ve = ValueDictionary(
            scoring_mechanism='any',
            aggregation_method=aggregation_method,
            preprocessing=preprocessing
        )
    elif value_estimation == "trained_bert":
        ve = ValueNetExtractor(checkpoint)
    else:
        raise ValueError("Unknown value estimation method")
    user2data = get_user2data(all_subreddits_posts)
    user2profile = ve.profile_multiple_users(user2data)
    dump_profiles = {k:v.tolist() for k,v in user2profile.items()}
    cfg_string = ""
    if len(preprocessing) > 0:
        cfg_string += "_".join(preprocessing)
    if aggregation_method is not None:
        cfg_string += f"_{aggregation_method}"
    if value_estimation == "trained_bert":
        cfg_string += f"_bert"

    dump_vectors(dump_profiles, f"data/user_values{cfg_string}.json")


def get_user_features(all_subreddits_posts, preprocessing=None):
    """
    Get the reddit user features for all users.
    """
    user2data = get_user2data(all_subreddits_posts)
    user2profile = {}
    for user in user2data:
        user_features_path = Path(f"output/reddit_userfeatures/{user}.json")
        if user_features_path.exists():
            with open(user_features_path, 'r') as f:
                user_features = json.load(f)
            user_vector = [
                float(user_features["comment_karma"]),
                float(user_features["link_karma"]),
                float(user_features["created_utc"]),
                float(user_features["is_gold"]),
                float(user_features["is_mod"]),
                float(user_features["is_employee"]),
                float(user_features["num_gilded"]),
                float(user_features["num_comments"]),
                float(user_features["num_submissions"])
            ]
            user2profile[user] = user_vector

        else:
            print(f"User {user} not found")
            user2profile[user] = [0] * 9
    filename_suffix = ""

    if preprocessing is not None:
        matrix = np.stack(list(user2profile.values()))
        users = list(user2profile.keys())
        if preprocessing  == "minmax":
            scaler = MinMaxScaler()
            filename_suffix = "_minmax"
        elif preprocessing == "standard":
            scaler = StandardScaler()
            filename_suffix = "_standard"
        profiles_scaled = scaler.fit_transform(matrix)
        user2profile = {}
        for i in range(matrix.shape[0]):
            user2profile[users[i]] = profiles_scaled[i, :].squeeze().tolist()
    dump_vectors(user2profile, f"data/user_features{filename_suffix}.json")


def get_noise(all_subreddits_posts, num_dims=768):
    """
    Get a random noise vector per user
    """
    user2data = get_user2data(all_subreddits_posts)
    user2noise = {}
    for user in user2data:
        user2noise[user] = np.random.rand(num_dims).tolist()
    dump_vectors(user2noise, f"data/user_noise.json")


def compute_kendall(left, right):
    """
    Compute kendall score.
    """
    ordered_left = np.argsort(left)
    ordered_right = np.argsort(right)
    profile_parent = 10 - ordered_left
    profile_child = 10 - ordered_right
    tau = kendalltau(profile_parent, profile_child).correlation
    return tau


def compute_absolute_error(left, right):
    """
    Compute absolute error. Normalize per value mention count individually to stick to relative
    error.
    """
    left = left / np.max(left)
    right = right / np.max(right)
    return np.sum(np.abs(left - right))


def compute_cos(left, right):
    score = cosine_similarity(left.reshape(1,-1), right.reshape(1,-1))
    return score[0][0]


def avg_coords(vector):
    """
    Average a bunch of vectors
    """
    M_left = np.zeros((len(vector),len(vector)))
    for i in range(len(vector)):
        for j in range(len(vector)):
            denom = np.sqrt(ValueConstants.SCHWARTZ_VALUE_SIMILARITY_MATRIX[i,j])
            M_left[i,j] = denom * ((vector[i] + vector[j]) / 2)
    return M_left


def compute_soft_cos(left, right):
    M_left = avg_coords(left)
    M_right = avg_coords(right)

    values = [ValueConstants.SCHWARTZ_VALUES[x] for x in ValueConstants.SCHWARTZ_VALUES_CIRCUMPLEX_ORDER]
    nominator = 0
    denom_left = 0
    denom_right = 0
    for i in range(len(values)):
        for j in range(len(values)):
            nominator += M_left[i,j] * M_right[i,j]
            denom_left += M_left[i,j] * M_left[i,j]
            denom_right += M_right[i,j] * M_right[i,j]
    return nominator / (np.sqrt(denom_left) * np.sqrt(denom_right))


def normalize_profile(profile, normalization_method, vd=None):
    """
    Normalize a single profile (np.array) .

    Apply either sum normalization or dictionary term frequency weighting.
    """
    if vd is None:
        vd = ValueDictionary(
            scoring_mechanism='any',
            aggregation_method='freqweight_normalized',
            preprocessing=['lemmatize'],
        )

    if normalization_method == "freqweight":
        prof = np.array(profile) / vd.word_freq_weight
        profile = vd.sum_normalize_profile(prof)
    elif normalization_method == "sum_normalize":
        profile = vd.sum_normalize_profile(profile)
    elif normalization_method == "minmax":
        scaler = MinMaxScaler()
        profile = np.array(profile)
        profile = scaler.fit_transform(profile.reshape(-1,1)).squeeze()
    return profile


def normalize_profiles(profiles, normalization_method):
    """
    Normalize all profiles in a dict (key is username).

    Apply either sum normalization or dictionary term frequency weighting.
    """
    vd = ValueDictionary(
        scoring_mechanism='any',
        aggregation_method='freqweight_normalized',
        preprocessing=['lemmatize'],
    )
    new_profiles = {}
    for user in profiles:
        new_profiles[user] = normalize_profile(profiles[user], normalization_method, vd=vd)
    return new_profiles

