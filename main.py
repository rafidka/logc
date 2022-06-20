# Python imports
import argparse
import fileinput
import json

# 3rd party imports
from sklearn.cluster import Birch, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pandas as pd

# logc imports
from tokenization import BertTokenizer, NltkTokenizer


# Define a map of tokenizers to their respective classes.
tokenizers = {
    'nltk': NltkTokenizer,
    'bert': BertTokenizer
}

# Define a map of clustering algorithms to their respective classes.
clusterers = {
    'kmeans': KMeans,
    'birch': Birch
}


def cluster_logs(logs: list[str], tokenizer, clusterer) -> dict[str, list[str]]:
    """
    Partition the given logs into clusters using the TF-IDF algorithm for
    vectorization along with the Birch algorithm for clustering.

    Keyword arguments:
    logs -- The logs to cluster.

    Returns:
    A dictionary mapping cluster numbers to lists of logs.
    """

    # Use TF-IDF algorithm to vectorize the input lines.
    vectors = TfidfVectorizer(
        tokenizer=tokenizer,
    ).fit_transform(tqdm(logs, desc='Iterating through logs to generate vector representations.'))

    # Use Birch to cluster the vector representations of the input lines.
    clusters = clusterer.fit_predict(vectors)

    # Create a dataframe with the input lines and their cluster labels, and
    # then sort by cluster.
    df = pd.DataFrame(zip(logs, clusters), columns=['log', 'cluster'])
    df = df.sort_values(by=['cluster'])

    # Returns a dictionary of cluster labels and their lines.
    return df.groupby('cluster')['log'].aggregate(list).to_dict()


def parse_args():
    parser = argparse.ArgumentParser(description='''
Clusters logs based on similarity.

The input is a list of logs, one per line. It can be read from stdin or
from files.
''')
    parser.add_argument(
        '-t',
        '--tokenizer',
        type=str,
        choices=['nltk', 'bert'],
        default='nltk',
        help='The tokenizer to use. Available options: nltk, bert.'
    )
    # Adds an argument for the clustering algorithm to use.
    parser.add_argument(
        '-c',
        '--clusterer',
        type=str,
        choices=['kmeans', 'birch'],
        default='kmeans',
        help='The clustering algorithm to use. Available options: kmeans, birch.'
    )
    # Adds an argument for the number of clusters to divide the log groups into.
    parser.add_argument(
        '-n',
        '--num-clusters',
        type=int,
        default=100,
        help='The number of clusters to divide the log groups into.'
    )
    # Adds an argument for the input files.
    parser.add_argument(
        'files',
        type=str,
        nargs='*',
        help='The input files to read logs from.'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    # Read input from stdin or files.
    # TODO Avoid reading all input into memory.
    input_logs = list(line.strip()
                      for line in fileinput.input(args.files) if line.strip() != '')

    # Creates the tokenizer and clusterer.
    tokenizer_cls = tokenizers[args.tokenizer]
    tokenizer = tokenizer_cls()
    clusterer_cls = clusterers[args.clusterer]
    clusterer = clusterer_cls(n_clusters=args.num_clusters)

    # Cluster the logs.
    clustered_logs = cluster_logs(input_logs, tokenizer, clusterer)

    print(json.dumps({
        "logClusters": [{
            "logs": logs
        } for _, logs in clustered_logs.items()]
    }, indent=2))


if __name__ == "__main__":
    main()
