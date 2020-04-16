from matplotlib import pyplot as plt


def np_hist_to_cv(np_histogram_output):
    counts, bin_edges = np_histogram_output
    return counts.ravel().astype('float32')


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)
