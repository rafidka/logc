# Python imports
import argparse
import fileinput

# 3rd party imports
from sklearn.cluster import Birch, KMeans
from tqdm import tqdm
import pandas as pd

# Local imports
from pat_cli.tokenization import NltkTokenizer, SimpleTokenizer
from pat_cli.vectorization import TfidfVectorizer, TfidfPlusWord2VecVectorizer


# Define a map of tokenizers to their respective classes.
tokenizers = {"simple": SimpleTokenizer, "nltk": NltkTokenizer}

# Define a map of vectorizers to their respective classes.
vectorizers = {
    "tfidf": TfidfVectorizer,
    "tfidf-word2vec": TfidfPlusWord2VecVectorizer,
}

# Define a map of clustering algorithms to their respective classes.
clusterers = {"kmeans": KMeans, "birch": Birch}


def cluster_logs(logs: list[str], vectorizer, clusterer) -> dict[str, list[str]]:
    """
    Partition the given logs into clusters using the given vectorizer and clusterer.

    Keyword arguments:
    logs -- The logs to cluster.
    vectorizer -- The vectorizer to use to convert the logs to vectors.
    clusterer -- The clusterer to use to cluster the vectors.

    Returns:
    A dictionary mapping cluster numbers to lists of logs.
    """

    # Converts the logs to vectors.
    vectors = vectorizer(logs)

    # Use Birch to cluster the vector representations of the input lines.
    clusters = clusterer.fit_predict(vectors)

    # Create a dataframe with the input lines and their cluster labels, and
    # then sort by cluster.
    df = pd.DataFrame(zip(logs, clusters), columns=["log", "cluster"])
    df = df.sort_values(by=["cluster"])

    # Returns a dictionary of cluster labels and their lines.
    return df.groupby("cluster")["log"].aggregate(list).to_dict()


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
Clusters logs based on similarity.

The input is a list of logs, one per line. It can be read from stdin or
from files.
"""
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        choices=list(tokenizers.keys()),
        default="nltk",
        help=f"The tokenizer to use. Available options: {list(tokenizers.keys())}.",
    )
    # Adds an argument for the clustering algorithm to use.
    parser.add_argument(
        "-c",
        "--clusterer",
        type=str,
        choices=["kmeans", "birch"],
        default="kmeans",
        help="The clustering algorithm to use. Available options: kmeans, birch.",
    )
    # Adds an argument for the number of clusters to divide the log groups into.
    parser.add_argument(
        "-n",
        "--num-clusters",
        type=int,
        default=100,
        help="The number of clusters to divide the log groups into.",
    )
    # Adds an argument for the vectorizer to use.
    parser.add_argument(
        "-v",
        "--vectorizer",
        type=str,
        choices=["tfidf", "tfidf-word2vec"],
        default="tfidf",
        help="""The vectorizer to use. Available options: tfidf, tfidf-word2vec.
        
tfidf: Uses the TF-IDF algorithm to vectorize the input lines.

tfidf-word2vec: Uses fasttext to vectorize the tokens of each line, and the
  TF-IDF as a mechanism for adding different weights to the different tokens
  making up the lines.
""",
    )
    # Adds an argument for the maximum number of logs to include in each cluster.
    parser.add_argument(
        "-m",
        "--max-logs-per-cluster",
        type=int,
        default=10,
        help="""The maximum number of logs to include in each cluster.
        
Usually, logs under the same cluster are similar in nature, so you don't need
to see all of them. As such, by default only 10 logs per cluster are
displayed. If you need more than that, you can increase this number. If you
need to see all of them, set this to 0.
""",
    )
    # Adds an argument for grep-like filtering.
    parser.add_argument(
        "-g",
        "--grep",
        type=str,
        help="Specify a pattern if you want to filter clusters.",
    )
    # Adds an argument for the input files.
    parser.add_argument(
        "files", type=str, nargs="*", help="The input files to read logs from."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Read input from stdin or files.
    # TODO Avoid reading all input into memory.
    input_logs = list(
        line.strip()
        for line in tqdm(fileinput.input(args.files), desc="Reading input...")
        if line.strip() != ""
    )

    # Creates the tokenizer, vectorizer, and clusterer.
    tokenizer_cls = tokenizers[args.tokenizer]
    tokenizer = tokenizer_cls()
    vectorizer_cls = vectorizers[args.vectorizer]
    vectorizer = vectorizer_cls(tokenizer)
    clusterer_cls = clusterers[args.clusterer]
    clusterer = clusterer_cls(
        n_clusters=args.num_clusters if args.num_clusters > 0 else None
    )

    # Cluster the logs.
    clustered_logs = cluster_logs(input_logs, vectorizer, clusterer)

    for _, logs in clustered_logs.items():
        if not logs:
            continue

        # Check if any log in the cluster matches the grep pattern.
        if args.grep:
            if not any(args.grep in log for log in logs):
                continue
            # Since there is a grep, we want to print only logs containing
            # the pattern.
            logs = [log for log in logs if args.grep in log]

        # Print the first log of each cluster unindented.
        print(logs[0])

        # Print the rest of the logs of each cluster indented.
        max_logs = (
            min(args.max_logs_per_cluster, len(logs) - 1)
            if args.max_logs_per_cluster > 0
            else len(logs) - 1
        )
        for log in logs[1:max_logs]:
            print(f"    {log}")

        print("\n\n")
