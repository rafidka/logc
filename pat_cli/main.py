# Python imports
import fileinput

# 3rd party imports
from tqdm import tqdm
from termcolor import colored
import pandas as pd

# Local imports
from pat_cli.args import clusterers, parse_args, tokenizers, vectorizers


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
        match_count = (
            colored(f" (match count: {count})", "green")
            if not args.hide_match_count
            else ""
        )
        header = colored(lines[0], "blue")
        print(f"{header}{match_count}")

        # Print the rest of the lines of each cluster indented.
        max_lines = (
            min(args.max_lines_per_cluster, len(lines) - 1)
            if args.max_lines_per_cluster > 0
            else len(lines) - 1
        )
        if not args.only_pattern:
            for line in lines[1:max_lines]:
                print(f"    {line}")

            print("\n\n")
