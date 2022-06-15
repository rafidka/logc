# Python imports
import fileinput

# nltk imports
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

# numpy et al. imports
import pandas as pd
from sklearn.cluster import Birch
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = PorterStemmer()


def tokenize(string: str) -> list[str]:
    """
    Split a string into multiple tokens.

    Keyword arguments:
    string -- The string to tokenize.

    Returns:
    A list of tokens.
    """
    tokens = word_tokenize(string)
    stemmed_tokens = [stemmer.stem(token)
                      for token in tokens if token.isalpha()]
    return stemmed_tokens


def cluster_logs(logs: list[str]) -> dict[str, list[str]]:
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
        tokenizer=tokenize,
        stop_words='english'
    ).fit_transform(logs)

    # Use Birch to cluster the vector representations of the input lines.
    clusters = Birch(n_clusters=None).fit_predict(vectors)

    # Create a dataframe with the input lines and their cluster labels, and
    # then sort by cluster.
    df = pd.DataFrame(zip(logs, clusters), columns=['log', 'cluster'])
    df = df.sort_values(by=['cluster'])

    # Returns a dictionary of cluster labels and their lines.
    return df.groupby('cluster')['log'].aggregate(list).to_dict()


def main():
    # Read input from stdin or files.
    # TODO Avoid reading all input into memory.
    input_logs = list(fileinput.input())

    clustered_logs = cluster_logs(input_logs)

    for cluster, logs in clustered_logs.items():
        print(f'[{cluster}]')
        for log in logs:
            print(f'  {log.strip()}')


if __name__ == "__main__":
    main()
