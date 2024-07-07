# Image Segmentation using Naive Bayes and Graph Cuts

This Python script demonstrates an image segmentation technique using naive Bayes classification and graph cuts. The process involves:

Building a graph from an image where foreground and background regions are determined using naive Bayes classifiers.
Computing probabilities for all pixels based on the trained classifiers.
Using graph cut algorithms to segment the image based on computed probabilities and pixel similarities.
Evaluating the segmentation accuracy using a predefined ground truth.
The main components of the script include:

* _SimpleBayesClassifier_: A class implementing a simple naive Bayes classifier for training and classification.
* _build_bayes_graph_: Function to construct a graph representation of the image with nodes representing pixels and edges representing relationships between adjacent pixels.
* _cut_graph_: Function to solve the maximum flow problem on the constructed graph and derive the binary labels for segmentation.
* _segmentation_error_: Function to calculate segmentation error based on a ground truth segmentation map.

## Setup

### Create virtual environment:

* python3 -m venv venv

### Activate the virtual environment:

On macOS/Linux:

* source venv/bin/activate

On Windows:
* venv\Scripts\activate

### Install dependencies:

* pip install -r requirement.txt