from sklearn.manifold import TSNE
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import yaml
from keras import optimizers
from embedding_net.augmentations import get_aug
from embedding_net.data_loader import EmbeddingNetImageLoader


def load_encodings(path_to_encodings):

    with open(path_to_encodings, 'rb') as f:
        encodings = pickle.load(f)
    return encodings


def plot_tsne(encodings_path, save_plot_dir, show=True):
    encodings = load_encodings(encodings_path)
    labels = list(set(encodings['labels']))
    tsne = TSNE()
    tsne_train = tsne.fit_transform(encodings['encodings'])
    fig, ax = plt.subplots(figsize=(16, 16))
    for i, l in enumerate(labels):
        xs = tsne_train[np.array(encodings['labels']) == l, 0]
        ys = tsne_train[np.array(encodings['labels']) == l, 1]
        ax.scatter(xs, ys, label=l)
        for x, y in zip(xs, ys):
            plt.annotate(l,
                         (x, y),
                         size=8,
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

    ax.legend(bbox_to_anchor=(1.05, 1), fontsize='small', ncol=2)
    if show:
        fig.show()

    fig.savefig("{}{}.png".format(save_plot_dir, 'tsne.png'))


def plot_tsne_interactive(encodings_path):
    import plotly.graph_objects as go
    encodings = load_encodings(encodings_path)
    labels = list(set(encodings['labels']))
    tsne = TSNE()
    tsne_train = tsne.fit_transform(encodings['encodings'])
    fig = go.Figure()
    for i, l in enumerate(labels):
        xs = tsne_train[np.array(encodings['labels']) == l, 0]
        ys = tsne_train[np.array(encodings['labels']) == l, 1]
        color = 'rgba({},{},{},{})'.format(int(255*np.random.rand()),
                                           int(255*np.random.rand()),
                                           int(255*np.random.rand()), 0.8)
        fig.add_trace(go.Scatter(x=xs,
                                 y=ys,
                                 mode='markers',
                                 marker=dict(color=color,
                                             size=10),
                                 text=l,
                                 name=l))
    fig.update_layout(
        title=go.layout.Title(text="t-SNE plot",
                              xref="paper",
                              x=0),
        autosize=False,
        width=1000,
        height=1000
    )

    fig.show()


def plot_grapths(history, save_path):
    for k, v in history.history.items():
        t = list(range(len(v)))
        fig, ax = plt.subplots()
        ax.plot(t, v)

        ax.set(xlabel='epoch', ylabel='{}'.format(k),
               title='{}'.format(k))
        ax.grid()

        fig.savefig("{}{}.png".format(save_path, k))


def parse_net_params(filename='configs/road_signs.yml'):
    params = {}
    with open(filename, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    learning_rate = cfg['learning_rate']
    if cfg['optimizer'] == 'adam':
        optimizer = optimizers.Adam(lr=learning_rate)
    elif cfg['optimizer'] == 'rms_prop':
        optimizer = optimizers.RMSprop(lr=learning_rate)
    elif cfg['optimizer'] == 'radam':
        from keras_radam import RAdam
        optimizer = RAdam(learning_rate)
    else:
        optimizer = optimizers.SGD(lr=learning_rate)

    if 'augmentations_type' in cfg:
        augmentations = get_aug(cfg['augmentation_type'], cfg['input_shape'])
    else:
        augmentations = None

    params = {k: v for k, v in cfg.items() if k not in ['optimizer']}

    if 'dataset_path' in cfg:
        params['loader'] = EmbeddingNetImageLoader(cfg['dataset_path'],
                                                   input_shape=cfg['input_shape'],
                                                   augmentations=augmentations)

    params['optimizer'] = optimizer
    return params



encodings_path='C:\\Users\\Quim\\Desktop\\EmbeddingNet\\work_dirs\\ai_city_challenge\\encodings\\encodings_without_training_ai_city_challenge_resnext50.pkl'
#plot_tsne_interactive(encodings_path)