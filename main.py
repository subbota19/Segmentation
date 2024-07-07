import networkx as nx
from numpy.linalg import norm as linalg_norm
from numpy import exp, zeros, sum as np_sum, array, logical_and, concatenate
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


class SimpleBayesClassifier:
    def __init__(self):
        self.model = GaussianNB()

    def train(self, train_data):
        X = concatenate(train_data)
        y = array([0] * len(train_data[0]) + [1] * len(train_data[1]))
        self.model.fit(X, y)

    def classify(self, data):
        prob = self.model.predict_proba(data)
        return self.model.predict(data), prob.T


def build_bayes_graph(im, labels, sigma=1e2, kappa=2):
    """ Build a graph from 4-neighborhood of pixels.
    Foreground and background is determined from
    labels (1 for foreground, -1 for background, 0 otherwise)
    and is modeled with naive Bayes classifiers."""
    m, n = im.shape[:2]

    # RGB vector version (one pixel per row)
    vim = im.reshape((-1, 3))
    # RGB for foreground and background
    foreground = im[labels == 1].reshape((-1, 3))
    background = im[labels == -1].reshape((-1, 3))
    train_data = [foreground, background]
    # train naive Bayes classifier
    bc = SimpleBayesClassifier()
    bc.train(train_data)
    # get probabilities for all pixels
    bc_labels, prob = bc.classify(vim)
    prob_fg = prob[1]
    prob_bg = prob[0]
    # create graph with m*n+2 nodes
    gr = nx.DiGraph()
    source = m * n  # second to last is source
    sink = m * n + 1  # last node is sink
    # normalize
    for i in range(vim.shape[0]):
        vim[i] = vim[i] / linalg_norm(vim[i])
    # go through all nodes and add edges
    for i in range(m * n):
        # add edge from source
        gr.add_edge(source, i, capacity=(prob_fg[i] / (prob_fg[i] + prob_bg[i])))
        # add edge to sink
        gr.add_edge(i, sink, capacity=(prob_bg[i] / (prob_fg[i] + prob_bg[i])))
        # add edges to neighbors
        if i % n != 0:  # left exists
            edge_wt = kappa * exp(-1.0 * np_sum((vim[i] - vim[i - 1]) ** 2) / sigma)
            gr.add_edge(i, i - 1, capacity=edge_wt)
        if (i + 1) % n != 0:  # right exists
            edge_wt = kappa * exp(-1.0 * np_sum((vim[i] - vim[i + 1]) ** 2) / sigma)
            gr.add_edge(i, i + 1, capacity=edge_wt)
        if i // n != 0:  # up exists
            edge_wt = kappa * exp(-1.0 * np_sum((vim[i] - vim[i - n]) ** 2) / sigma)
            gr.add_edge(i, i - n, capacity=edge_wt)
        if i // n != m - 1:  # down exists
            edge_wt = kappa * exp(-1.0 * np_sum((vim[i] - vim[i + n]) ** 2) / sigma)
            gr.add_edge(i, i + n, capacity=edge_wt)
    return gr


def cut_graph(gr, imsize):
    """ Solve max flow of graph gr and return binary
    labels of the resulting segmentation."""
    m, n = imsize
    source = m * n  # second to last is source
    sink = m * n + 1  # last is sink
    # cut the graph
    flow_value, flows = nx.maximum_flow(gr, source, sink)
    # get the segments
    cut_value, partition = nx.minimum_cut(gr, source, sink)
    reachable, non_reachable = partition
    # convert graph to image with labels
    res = zeros(m * n)
    for node in reachable:
        if node != source:
            res[node] = 1
    return res.reshape((m, n))


def segmentation_error(ground_truth, segmentation):
    """ Calculate the segmentation error based on ground truth and the segmentation result. """
    error = np_sum(logical_and(ground_truth != 0, ground_truth != segmentation))
    total = np_sum(ground_truth != 0)
    return 100 * (error / total)


def main():
    im = Image.open('pic/empire')
    im = im.resize((im.width // 14, im.height // 14), Image.BILINEAR)
    im = array(im)
    size = im.shape[:2]
    # add two rectangular training regions
    labels = zeros(size)
    labels[3:18, 3:18] = -1
    labels[-18:-3, -18:-3] = 1
    # create graph
    g = build_bayes_graph(im, labels, kappa=1)
    # cut the graph
    res = cut_graph(g, size)

    # Load the ground truth segmentation map
    ground_truth = array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1]
    ])

    # Calculate segmentation error
    error = segmentation_error(ground_truth, res)
    print(f"Segmentation Error: {error}%")

    # Display results
    plt.figure()
    plt.imshow(im)
    plt.title('Original Image')

    plt.figure()
    plt.imshow(res, cmap='gray')
    plt.title('Segmentation Result')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
