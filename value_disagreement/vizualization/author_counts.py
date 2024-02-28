import matplotlib.pyplot as plt


def plot_subreddit_author_counts(cc, subreddit):
    """
    Plot the comment counts of authors per subreddit.
    """

    # Sort by frequency and plot
    sorted_cc = cc.most_common(n=None)
    xs = [x for x, _ in sorted_cc]
    ys = [y for _, y in sorted_cc]
    plt.figure(figsize=(12, 4), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(xs, ys)
    plt.xticks(rotation=90, fontsize=1)
    plt.savefig(f"output/author_counts_{subreddit}.png")
    return set(cc.keys())