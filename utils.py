from os import path
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib import patches
from matplotlib.collections import PatchCollection
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
from sklearn import preprocessing
from sklearn.utils.linear_assignment_ import linear_assignment

def invert_permutation(p, permutated_array):
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return permutated_array[s]


def permute(X, Y=None):
    p = np.random.permutation(X.shape[0])
    X = X[p]
    if Y is not None:
        Y =Y[p]
    return p, X,Y


def read_data(file_path, seperator=',', has_labels=True):
    if path.split[1] == '.arff':
        return read_arff(file_path, seperator=seperator, has_labels=has_labels)
    else:
        return read_csv_data(file_path, seperator=seperator, has_labels=has_labels)


def read_csv_data(file_path, seperator=',', has_labels=True):
    with open(file_path) as handle:
        data = []
        labels = None
        if has_labels:
            labels = []
        
        for line in handle:
            line = line.rstrip()
            if len(line) == 0:
                continue
            row = []
            line_parts = line.split(seperator)
            row = [float(i) for i in line_parts[:len(line_parts)-1]]
            data.append(row)
            if has_labels:
                label = int(line_parts[-1])
                labels.append(label)
        
        if has_labels:
            return np.ndarray(shape=(len(data), len(data[0])), buffer=np.matrix(data)), np.array(labels)
        else:
            return np.ndarray(shape=(len(data), len(data[0])), buffer=np.matrix(data)), None


def read_arff(file_path, seperator=",", has_labels=True):
    read_data = False
    data  = []
    labels = None
    if (has_labels):
        labels = []
        
    with open(file_path) as handle:
        for l in handle:
            l = l.rstrip()
            if l == "":
                continue
            if (read_data):
                splitted = l.split(seperator)

                if (has_labels):
                    row = [float(s) for s in splitted]
                else:
                    row = [float(s) for s in splitted[:len(splitted)-1]]
                    labels.append(splitted[len(splitted)-1])
                data.append(row)
                
            elif (l.lower() == "@data"):
                read_data = True
    
    le = preprocessing.LabelEncoder()
    encoded_labels = None
    if has_labels:
        encoded_labels = np.array(le.fit_transform(labels))
    return np.ndarray(shape=(len(data), len(data[0])), buffer=np.matrix(data)), encoded_labels

def save_data(file_path, data , labels):
    with open(file_path, "w") as handle:
        for p, l in zip(data, labels):
            line = ",".join([str(s) for s in p]) + "," + str(l)
            handle.write(line + "\n")

def save_labels(file_path, labels):
    with open(file_path, "w") as handle:
        for l in labels:
            handle.write(str(l) + "\n")

def load_from_file_or_data(obj, seperator=',', dim=2, hasLabels=False):
    if (type(obj) is str):
        return read_data(obj, seperator=seperator, dim=dim, hasLabels=hasLabels)
    else:
        return obj


def to_original_clusters(y_pred, y_true):
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    y_pred_translated = np.zeros(len(y_pred))
    for i,y in enumerate(y_pred):
        y_pred_translated[i] = ind[y_pred[i]][1]
    return y_pred_translated.astype(int)

def draw_clusters(X, labels, colors=None, show_plt=True, show_title=False, name=None, ax=None,
                  markersize=15, markeredgecolor='k', use_clustes_as_keys = False, linewidth=0,
                  noise_data_color='k'):
    import seaborn as sns
    if (ax == None):
        ax = plt
    #unique_labels = set(labels)
    unique_labels = np.unique(labels)
    label_map = sorted(unique_labels)
    if (colors == None):
        colors = sns.color_palette()
        if len(colors) < len(unique_labels):
            colors = plt.cm.Spectral(np.linspace(1, 0, len(unique_labels)))
    has_noise = False

    if not use_clustes_as_keys:
        if (label_map[0] == -1):
            if (isinstance(colors, list)):
                colors = [noise_data_color] + colors
            else:
                colors = [noise_data_color] + colors.tolist()

    #for k, col in zip(label_map, colors):
    for k, i in zip(label_map, xrange(len(label_map))):
        if k == -1:
            # Black used for noise.
            col = noise_data_color
            has_noise = True
        else:
            if use_clustes_as_keys:
                col = colors[int(k)]
            else:
                col = colors[i]
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], s=markersize, facecolor=col,
                 edgecolor=markeredgecolor, linewidth=linewidth)
        #ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
        #         markeredgecolor=markeredgecolor, markersize=markersize, lw=lw)

    if (show_title):
        labels_count = len(unique_labels)
        if (has_noise):
            labels_count = labels_count - 1
        title_prefix = ""
        if (name != None):
            title_prefix = "%s - "%name
        if hasattr(ax, 'set_title'):
            ax.set_title((title_prefix + 'Estimated number of clusters: %d') % len(unique_labels))
        else:
            ax.title((title_prefix + 'Estimated number of clusters: %d') % len(unique_labels))
    #if (show_plt):
    #    ax.show()
    return ax


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_clusters3d(X, labels, colors=None, show_plt=True, show_title=False, name=None, ax=None, markersize=15, markeredgecolor='k', linewidth=0):
    import seaborn as sns
    #if (ax == None):
    #    ax = plt
    #unique_labels = set(labels)
    fig = plt.figure(figsize=(float(1600) / float(72), float(1600) / float(72)))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    label_map = sorted(unique_labels)
    if (colors == None):
        colors = sns.color_palette()
        #colors = plt.cm.Spectral(np.linspace(1, 0, len(unique_labels)))
    has_noise = False

    if (label_map[0] == -1):
        colors = ['k'] + colors

    for k, col in zip(label_map, colors):
        if k == -1:
            # Black used for noise.
            #col = 'k'
            has_noise = True

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=markersize, c=col)
#                 edgecolor=markeredgecolor, linewidth=linewidth)
        #ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
        #         markeredgecolor=markeredgecolor, markersize=markersize, lw=lw)

    if (show_title):
        labels_count = len(unique_labels)
        if (has_noise):
            labels_count = labels_count - 1
        title_prefix = ""
        if (name != None):
            title_prefix = "%s - "%name
        if hasattr(ax, 'set_title'):
            ax.set_title((title_prefix + 'Estimated number of clusters: %d') % len(unique_labels))
        else:
            ax.title((title_prefix + 'Estimated number of clusters: %d') % len(unique_labels))
    #if (show_plt):
    #    ax.show()
    #ax.set_zlim([-0.01, 0])
    return ax
