from keras_dec import DeepEmbeddingClustering
from keras.datasets import mnist
import numpy as np
import argparse
import utils

def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis = 0)
    Y = np.concatenate((y_train, y_test), axis = 0)
    X = x_all.reshape(-1,x_all.shape[1]*x_all.shape[2]) * 0.02
    return X, Y


parser = argparse.ArgumentParser(description='DEC Clustering with Keras')
parser.add_argument('--input', type=str, metavar='<file path>',
                   help='Path to comma separated input file (or .arff), unless specified otherwise, last column is ground truth label. If not specified MNIST is used', required=False)
parser.add_argument('--output', type=str, metavar='<file path>',
                    help='Path to output file, which contains the clustering results', required=False)
parser.add_argument("--no-labels", help="Specify that input file has no ground truth labels (default is false)",
                    action="store_true")
parser.add_argument('--encoder-weights-input-path', type=str, metavar='<file path>',
                    help='Path to a file that contains the autoencoder weights', required=False)
parser.add_argument('--encoder-weights-output-path', type=str, metavar='<file path>',
                    help='The path of the file in which the autoencoder weights will be saved', required=False)
parser.add_argument('--max-cluster-iterations', type=float, metavar='<number of iterations>',
                    help='The number of training iterations for clustering', required=False, 
                    default=1e6)


args = parser.parse_args()

output_file_path = args.output
input_file_path =  args.input
input_has_labels = not args.no_labels
encoder_weights_path = args.encoder_weights_input_path
encoder_weights_output_path = args.encoder_weights_output_path
max_number_of_cluster_iterations = args.max_cluster_iterations

if encoder_weights_path is not None and encoder_weights_output_path is not None:
    parser.error("Cannot provider both input and output path for autoencoder weights")

if input_file_path is not None:
    X_original, Y_original = utils.read_data(input_file_path, has_labels=input_has_labels)
else:
    X_original, Y_original  = get_mnist()

# permute the data to make sure that the batches randomized (TODO: make sure this is needed?)
np.random.seed(1234) # set seed for deterministic ordering
p, X, Y = utils.permute(X_original, Y_original)
n_clusters = len(np.unique(Y))
c = DeepEmbeddingClustering(n_clusters=n_clusters, input_dim=X.shape[1], pretrained_weights_path=encoder_weights_path)
c.initialize(X, finetune_iters=100000, layerwise_pretrain_iters=50000, autuencoder_weights_save_path=encoder_weights_output_path)
clusters_permutated = c.cluster(X, y=Y, iter_max=max_number_of_cluster_iterations)
clusters = utils.invert_permutation(p, clusters_permutated)


if output_file_path is not None:
    utils.save_data(output_file_path, utils.invert_permutation(p, X / 0.02), clusters)
    print("Saved clustering result to %s"%output_file_path)



