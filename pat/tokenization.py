import re


class BaseTokenizer:
    def __call__(self, string):
        """
        Tokenize the given string.

        Keyword arguments:
        string -- The string to tokenize.

        Returns:
        A list of tokens.
        """
        return self.tokenize(string)

    def tokenize(self, string):
        raise NotImplementedError

    # Inherit a documentation for `tokenize` from the `__call__` method.
    tokenize.__doc__ = __call__.__doc__


class SimpleTokenizer(BaseTokenizer):
    """
    A simple tokenizer for splitting log lines by whitespaces, symbols, numbers,
    numbers, and by camel case.
    """

    def tokenize(self, string: str) -> list[str]:
        # Split on whitespaces and symbols
        tokens = re.split(r"[\s,\._-]+", string)

        # Break down camel case tokens
        new_tokens = []
        for token in tokens:
            new_tokens.extend(re.findall(r"[A-Z][a-z]*", token))

        if len(new_tokens) == 0:
            return tokens
            # raise ValueError(f'Could not tokenize the string: {string}')

        return new_tokens


class NltkTokenizer(BaseTokenizer):
    """
    A tokenizer that uses the NLTK library to tokenize a string.

    Example:
    >>> tokenizer = NltkTokenizer()
    >>> tokenizer("Hello, world!")
    ['Hello', ',', 'world', "!"]
    """

    def __init__(self):
        """
        Initializes the tokenizer.
        """
        try:
            from nltk import word_tokenize
            from nltk.stem.porter import PorterStemmer

            self.nltk_word_tokenize = word_tokenize
            self.stemmer = PorterStemmer()
        except ImportError:
            raise ImportError(
                "NLTK is not installed. Please install it by running: "
                "pip install nltk"
            )

    def tokenize(self, string: str) -> list[str]:
        tokens = self.nltk_word_tokenize(string)
        stemmed_tokens = [
            self.stemmer.stem(token) for token in tokens if token.isalpha()
        ]
        return stemmed_tokens
