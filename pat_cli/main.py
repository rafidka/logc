# Python imports
import argparse
import fileinput

# 3rd party imports
from sklearn.cluster import Birch, KMeans
from tqdm import tqdm
from termcolor import colored
import pandas as pd

# Local imports
from . import __version__
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


def cluster_lines(lines: list[str], vectorizer, clusterer) -> dict[str, list[str]]:
    """
    Partition the given lines into clusters using the given vectorizer and clusterer.

    Keyword arguments:
    lines -- The lines to cluster.
    vectorizer -- The vectorizer to use to convert the lines to vectors.
    clusterer -- The clusterer to use to cluster the vectors.

    Returns:
    A dictionary mapping cluster numbers to lists of lines.
    """

    # Converts the lines to vectors.
    vectors = vectorizer(lines)

    # Use Birch to cluster the vector representations of the input lines.
    clusters = clusterer.fit_predict(vectors)

    # Create a dataframe with the input lines and their cluster labels, and
    # then sort by cluster.
    df = pd.DataFrame(zip(lines, clusters), columns=["line", "cluster"])
    df = df.sort_values(by=["cluster"])

    # Returns a dictionary of cluster labels and their lines.
    return df.groupby("cluster")["line"].aggregate(list).to_dict()


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
Clusters lines based on similarity.

The input is a list of lines, one per line. It can be read from stdin or
from files.
"""
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        choices=list(tokenizers.keys()),
        default="simple",
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
    # Adds an argument for the number of clusters to divide the line groups into.
    parser.add_argument(
        "-n",
        "--num-clusters",
        type=int,
        default=100,
        help="The number of clusters to divide the line groups into.",
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
    # Adds an argument for the maximum number of lines to include in each cluster.
    parser.add_argument(
        "-m",
        "--max-lines-per-cluster",
        type=int,
        default=10,
        help="""The maximum number of lines to include in each cluster.
        
Usually, lines under the same cluster are similar in nature, so you don't need
to see all of them. As such, by default only 10 lines per cluster are
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
        "files", type=str, nargs="*", help="The input files to read lines from."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Read input from stdin or files.
    # TODO Avoid reading all input into memory.
    input_lines = list(
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

    # Cluster the lines.
    clustered_lines = cluster_lines(input_lines, vectorizer, clusterer)

    for _, lines in sorted(
        clustered_lines.items(), key=lambda x: len(x[1]), reverse=True
    ):
        if not lines:
            continue

        count = len(lines)

        # Check if any line in the cluster matches the grep pattern.
        if args.grep:
            if not any(args.grep in line for line in lines):
                continue
            # Since there is a grep, we want to print only lines containing
            # the pattern.
            lines = [line for line in lines if args.grep in line]

        # Print the first line of each cluster unindented.
        match_count = colored(f"(match count: {count})", "green")
        header = colored(lines[0], "blue")
        print(f"{header} {match_count}")

        # Print the rest of the lines of each cluster indented.
        max_lines = (
            min(args.max_lines_per_cluster, len(lines) - 1)
            if args.max_lines_per_cluster > 0
            else len(lines) - 1
        )
        for line in lines[1:max_lines]:
            print(f"    {line}")

        print("\n\n")
