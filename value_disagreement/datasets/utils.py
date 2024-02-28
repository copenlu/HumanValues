import gensim
import numpy as np
from datasets import Dataset, Value
from sklearn.cluster import MiniBatchKMeans


def text2vectors(text: list, w2v_model, maxlen: int, vocabulary):
    """
    Token sequence -- to a list of word vectors;
    if token not in vocabulary, it is skipped; the rest of
    the slots up to `maxlen` are replaced with zeroes

    :param text: list of tokens
    :param w2v_model: gensim w2v model
    :param maxlen: max. length of the sentence; the rest is just cut away
    :return:
    """

    acc_vecs = []

    hits, misses = 0, 0
    for word in text:
        if word in w2v_model.wv and (vocabulary is None or word in vocabulary):
            hits += 1
            acc_vecs.append(w2v_model.wv[word])
        else:
            misses += 1

    # padding for consistent length with ZERO vectors
    if len(acc_vecs) < maxlen:
        acc_vecs.extend([np.zeros(w2v_model.vector_size)] * (maxlen - len(acc_vecs)))

    return acc_vecs, hits, misses


def get_w2v(path):
    """
    Reading word2vec model given the path.
    """
    return gensim.models.Word2Vec.load(str(path))


def get_centroids(w2v_model, aspects_count):
    """
    Clustering all word vectors with K-means and returning L2-normalizes
    cluster centroids; used for ABAE aspects matrix initialization
    """

    km = MiniBatchKMeans(n_clusters=aspects_count, verbose=0, n_init=100)
    m = []

    for k in w2v_model.wv.key_to_index:
        m.append(w2v_model.wv[k])

    m = np.asarray(m)
    km.fit(m)
    clusters = km.cluster_centers_

    # L2 normalization
    norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)

    return norm_aspect_matrix


def cast_dataset_to_hf(dataset, split_name, dataset_type="pandas", load_context=None, context_append_mode=None, abs_label=True):
    """
    Convert custom dataset to huggingface dataset.
    """
    if dataset_type == "pandas":
        if abs_label:
            labels = dataset['orig_label'] = [abs(x['orig_label']) for _, x in dataset.iterrows()]
        else:
            labels = dataset['orig_label'] = [x['orig_label'] for _, x in dataset.iterrows()]
        dataset_dict = {
            'id': list(range(len(dataset))),
            'text': [x['text'] for _, x in dataset.iterrows()],
            'orig_label': labels,
            'value': [x['value'] for _, x in dataset.iterrows()],
        }
    elif dataset_type == "agreement":
        dataset_dict = {
            'id': list(range(len(dataset))),
            'text_left': [x[0] for x in dataset],
            'text_right': [x[1] for x in dataset],
            'labels': [x[2] for x in dataset],
        }

    if load_context is not None:
        extra_info_train = [extra_info for _,_,_,extra_info in dataset]
        parent_vectors = [np.array(load_context[x['author_parent']], dtype=np.double) for x in extra_info_train]
        child_vectors = [np.array(load_context[x['author_child']], dtype=np.double) for x in extra_info_train]
        dataset_dict['author_left_vectors'] = parent_vectors
        dataset_dict['author_right_vectors'] = child_vectors
        if context_append_mode == "mixin_token":
            dataset_dict['sim_scores'] = [x['sim_scores'] for x in extra_info_train]

    hf_dataset = Dataset.from_dict(dataset_dict, split=split_name)
    return hf_dataset


def hf_dataset_tokenize(hf_dataset, tokenizer, model_name, context=None, context_append_mode=None, soft_target_type='int'):
    """
    Apply a tokenizer to a huggingface dataset.
    """
    tokenized_dataset = hf_dataset.map(tokenizer.tokenize, batched=True)

    column_names = ['input_ids', 'attention_mask', 'labels']
    if 'roberta' not in model_name:
        column_names.append("token_type_ids")
    if context is not None:
        if context_append_mode == "concat_vector":
            column_names.append("user_context")
        elif context_append_mode == "mixin_token":
            column_names.append("sim_scores")

    tokenized_dataset.set_format(type='torch', columns=column_names)
    if soft_target_type == 'float':
        target_type = Value(dtype='float32', id=None)
    else:
        target_type = Value(dtype='int64', id=None)
    tokenized_dataset = tokenized_dataset.cast_column('labels', target_type)
    return tokenized_dataset
