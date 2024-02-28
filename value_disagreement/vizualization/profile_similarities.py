from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def prepare_plot_data(label2sims):
    """
    Prepares the data for plotting in plt functions, by removing NaN values.
    """
    labels, data = label2sims.keys(), label2sims.values()
    data_arr = [np.array(x) for x in list(data)]
    filtered_data = [x[~np.isnan(x)] for x in data_arr]
    return labels, filtered_data


def get_title(args):
    """
    Returns the title for the plot based on subreddit and mimimum profile sum.
    """
    return f"{args.subreddit}, min_sum={args.profile_min_sum}, BF10={float(args.BF):.3f}"


def get_ylabel(args):
    """
    Returns the y-axis label for the plot based on similarity method.
    """
    if args.similarity_method == "kendall":
        return "Kendall's tau ($\\uparrow$)"
    elif args.similarity_method == "absolute_error":
        return "Absolute error ($\downarrow$)"
    elif args.similarity_method == "cosine_sim":
        return "Cosine similarity ($\\uparrow$)"
    elif args.similarity_method == "schwartz_soft_cosine":
        return "Soft cosine similarity ($\\uparrow$)"


def get_xticks(labels):
    """
    Returns the xticks for the plot based on the labels.
    """
    xticks = range(1, len(labels) + 1)
    return xticks


def annotate_plot(xticks, data):
    """
    Annotates the plot with the number of datapoints per label
    """
    X_OFFSET = -0.075
    Y_OFFSET = 0.03
    for i, tick in enumerate(xticks):
        plt.text((tick/len(xticks)) + (X_OFFSET * (i+1)), Y_OFFSET,
                 f"count={len(list(data)[i])}", transform=plt.gcf().transFigure,
                 horizontalalignment='center')


def get_output_filename(args, plot_type, prefix="output/", filetype="png"):
    """
    Get the name of the output file based on the arguments.
    """
    prof_method = args.profiling_method
    if args.profile_processing is not None:
        prof_method += f"_{args.profile_processing}"
    return f"{prefix}{plot_type}_{prof_method}_{args.subreddit}_{args.profile_min_sum}_{args.similarity_method}.{filetype}"


def get_yrange(similarity_method):
    """
    Get the yrange for the plot based on the similarity method.
    """
    if similarity_method == "kendall":
        return (-1, 1.1)
    elif similarity_method == "absolute_error":
        return None
    elif similarity_method == "cosine_sim":
        return (0, 1.1)
    elif similarity_method == "schwartz_soft_cosine":
        return (0, 1.1)
    else:
        return None


def prepare_figure(args):
    """
    All the preparation for the figure, such as title, y-axis label, etc.
    """
    plt.figure()
    title = get_title(args)
    plt.title(title, fontsize=10)
    ylabel = get_ylabel(args)
    plt.ylabel(ylabel)

def finish_figure(labels, filtered_data):
    """
    Create xticks and annotate image with data counts.
    """
    xticks = get_xticks(labels)
    plt.xticks(xticks, labels)
    annotate_plot(xticks, filtered_data)


def box_plot_correlations(label2sims, args):
    """
    Create a box plot of the similarity scores for profiles per label.
    """
    prepare_figure(args)
    labels, filtered_data = prepare_plot_data(label2sims)
    plt.boxplot(filtered_data)
    finish_figure(labels, filtered_data)
    outname = get_output_filename(args, "box")
    plt.savefig(outname)


def violin_plot_correlations(label2sims, args):
    """
    Create a violin plot of the similarity scores for profiles per label.
    """
    prepare_figure(args)
    labels, filtered_data = prepare_plot_data(label2sims)
    plt.violinplot(filtered_data)
    plt.ylim(get_yrange(args.similarity_method))
    finish_figure(labels, filtered_data)
    outname = get_output_filename(args, 'violin')
    plt.savefig(outname)


def plot_profile_matrix(profiles, sim_fn):
    authors = list(profiles.keys())
    num_authors = len(authors)
    scores = np.zeros((num_authors, num_authors))
    author_pairs = []
    for i, author in enumerate(tqdm(authors)):
        for j, author_2 in enumerate(authors[:i]):
            score = sim_fn(profiles[author], profiles[author_2])
            scores[i][j] = score
            author_pairs.append((author, author_2, score))

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(scores)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(num_authors), labels=authors)
    ax.set_yticks(np.arange(num_authors), labels=authors)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im, ax=ax)
    ax.set_title("Similarity scores between author profiles")
    plt.savefig(f"output/profile_correlation_{sim_fn.__name__}.png")