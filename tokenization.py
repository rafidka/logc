class BaseTokenizer:
    def __call__(self, string):
        """
        Tokenize the given stirng.

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
        from nltk import word_tokenize
        from nltk.stem.porter import PorterStemmer

        self.nltk_word_tokenize = word_tokenize
        self.stemmer = PorterStemmer()

    def tokenize(self, string: str) -> list[str]:
        tokens = self.nltk_word_tokenize(string)
        stemmed_tokens = [self.stemmer.stem(token)
                          for token in tokens if token.isalpha()]
        return stemmed_tokens


class BertTokenizer(BaseTokenizer):
    """
    A tokenizer that uses the BERT uncased tokenizer to tokenize a string.

    Example:
    >>> tokenizer = BertTokenizer()
    >>> tokenizer("Hello, world!")
    ['hello', ',', 'world', "!"]
    >>> tokenizer('Conceptualize')
    ['conceptual', '##ize']
    """

    def __init__(self):
        """
        Initializes the tokenizer.
        """
        from transformers import BertTokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')

    def tokenize(self, string: str) -> list[str]:
        tokens = self.bert_tokenizer.tokenize(string)
        return tokens
