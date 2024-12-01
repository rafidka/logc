# Python imports
import argparse

# 3rd party imports
from sklearn.cluster import Birch, KMeans

# Local imports
from pat_cli import __version__
from pat_cli.tokenization import NltkTokenizer, SimpleTokenizer
from pat_cli.vectorization import TfidfPlusWord2VecVectorizer, TfidfVectorizer

# Define a map of tokenizers to their respective classes.
tokenizers = {"simple": SimpleTokenizer, "nltk": NltkTokenizer}


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
Clusters lines based on similarity.

The input is a list of lines, one per line. It can be read from stdin or
from files.
""".strip()
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
        help="""
The vectorizer to use. Available options: tfidf, tfidf-word2vec.
        
tfidf: Uses the TF-IDF algorithm to vectorize the input lines.

tfidf-word2vec: Uses fasttext to vectorize the tokens of each line, and the
  TF-IDF as a mechanism for adding different weights to the different tokens
  making up the lines.
""".strip(),
    )
    # Adds an argument for the maximum number of lines to include in each cluster.
    parser.add_argument(
        "-l",
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
        help="""
Specify a pattern if you want to filter clusters.

The pattern is applied to **all** lines in the cluster, rather than just the
first (the header). If the pattern is found in any of the lines, the cluster is
displayed. If the pattern is not found in any of the lines, the cluster is
skipped.

So, if the -p/--only-pattern flag is set, the header that is displayed might not
actually be matching the grep itself, but the cluster contains at least one line
that does match the pattern.
""".strip(),
    )
    # Adds an argument for the input files.
    parser.add_argument(
        "files", type=str, nargs="*", help="The input files to read lines from."
    )
    # Adds an argument for grep-like filtering.
    parser.add_argument(
        "-p",
        "--only-pattern",
        action="store_true",
        help="""
If specified, only shows patterns without showing samples similar lines. Default value
is False.
""".strip(),
    )

    # Adds an argument for grep-like filtering.
    parser.add_argument(
        "-m",
        "--hide-match-count",
        action="store_true",
        help="""
If specified, show the count of lines in each cluster. Default value is True.  You might
want to disable this if you want to do some post processing on cluster headers and
require purely just the text without any addition.
""".strip(),
    )

    return parser.parse_args()


# Define a map of vectorizers to their respective classes.
vectorizers = {
    "tfidf": TfidfVectorizer,
    "tfidf-word2vec": TfidfPlusWord2VecVectorizer,
}
# Define a map of clustering algorithms to their respective classes.
clusterers = {"kmeans": KMeans, "birch": Birch}
