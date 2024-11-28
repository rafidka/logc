from typing import Iterable

import numpy as np


class TfidfVectorizer:
    """
    Converts a corpus into vectors using the TF-IDF algorithm. The process is
    composed of two steps. First, the corpus is tokenized using the given
    tokenizer. Then, the tokens are converted into vectors using the TF-IDF
    algorithm.

    More information about the TF-IDF algorithm can be found in the Wikipedia
    page:

    https://en.wikipedia.org/wiki/Tf%E2%80%93idf

    """

    def __init__(self, tokenizer):
        """
        Initializes the TF-IDF vectorizer.

        Keyword arguments:
        tokenizer -- The tokenizer to use.
        """
        from sklearn.feature_extraction.text import (
            TfidfVectorizer as SklearnTfidfVectorizer,
        )

        self.tfidf_vectorizer = SklearnTfidfVectorizer(tokenizer=tokenizer)

    def __call__(self, corpus: Iterable[str]) -> np.ndarray:
        # TODO Can we use fit_transform multiple times, or we need to define
        # a new instance every time?
        return self.tfidf_vectorizer.fit_transform(corpus)


class TfidfPlusWord2VecVectorizer:
    """
    This is similar to the TfidfVectorizer, but it uses a word2vec model after
    the TF-IDF process to convert the k-dimensional vectors, where k is the
    number of tokens, into an n-dimensional vector, where n is the embedding size
    of the word2vec model.
    """

    def __init__(self, tokenizer):
        """
        Initializes the vectorizer.

        Keyword arguments:
        tokenizer -- The tokenizer to use.
        """
        from sklearn.feature_extraction.text import (
            TfidfVectorizer as SklearnTfidfVectorizer,
        )

        self.tfidf_vectorizer = SklearnTfidfVectorizer(tokenizer=tokenizer)
        import fasttext.util

        fasttext.util.download_model("en", if_exists="ignore")
        self.fasttext = fasttext.load_model("cc.en.300.bin")

    def __call__(self, corpus: Iterable[str]) -> np.ndarray:
        # TODO Can we use fit_transform multiple times, or we need to define
        # a new instance every time?
        tfidf_vectors = self.tfidf_vectorizer.fit_transform(corpus)

        feature_vectors = np.array(
            [
                self.fasttext.get_word_vector(word)
                for word in self.tfidf_vectorizer.get_feature_names_out()
            ]
        )

        return np.matmul(tfidf_vectors.toarray(), feature_vectors)
