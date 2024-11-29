# pat-cli

## Overview

`pat-cli` is a tool for clustering logs based on the textual content of the log.
The tool uses a two-step process to achieve this:

**Vectorization**: In this step, the log statement is converted into a vector in
an n-dimensional space. This way, we can treat logs just like we treat points on
a 2D graph and try to cluster them. The only difference is that the dimension of
this space.

To achieve this vectorization, few algorithms can be employed. The available
algorithms can be found using the help page of the tool, but one example is the
[TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

Usually, vectorization involves a sub-step called **tokenization**. In this
step, the logs are broken down into a set of tokens. For example, a log like
"Writing output to file" can be broken down into the following tokens:
"writing", "output", "to", "file". This allows the tool to understand the
textual content of the logs. For example, in the TF-IDF algorithm, the
frequencies of the words (tokens) making up each log statement are used to
determine how important each word in the log is.

**Clustering**: In this step, a clustering algorithm like K-Means or Birch
is used to cluster the logs into multiple groups that are likely to be similar
to each other.

## Installation

First, clone the code locally:

```
git clone https://github.com/rafidka/pat-cli.git
```

After that, you need to install the Python requirements. It is recommended that
you create a virtual or Conda environment.

```
pip install -r requirements.txt
```

Finally, run the help page to learn how to use the tool:

```
python main.py --help
```

## Issue

If you create any issue, feel free to create a GitHub Issue in this repository
and I will try to address it or respond to it as soon as possible.

## Contribution

Contribution is welcome. If you have an interesting addition to the tool, be it
another vectorization or clustering algorithm, feel free to publish a PR.
