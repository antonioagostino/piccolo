from abc import ABC, abstractmethod

import tiktoken


class Tokenizer(ABC):
    """
    Abstract base class for tokenizers.
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Encode a text string into a list of token IDs.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list[int]: A list of token IDs.
        """

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.

        Args:
            tokens (list[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """

    @abstractmethod
    def get_end_token(self) -> int:
        """
        Return the token ID that represents the end of a sequence.

        Returns:
            int: The end-of-sequence token ID.
        """


class TiktokenTokenizer(Tokenizer):
    """
    Tokenizer implementation wrapping the tiktoken library."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialise the TiktokenTokenizer with a specified encoding.

        Args:
            encoding_name (str): The name of the tiktoken encoding to use.
        """
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        """
        Encode a text string into a list of token IDs.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list[int]: A list of token IDs.
        """
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.

        Args:
            tokens (list[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(tokens)

    def get_end_token(self) -> int:
        """
        Return the token ID that represents the end of a sequence.

        Returns:
            int: The end-of-sequence token ID.
        """
        return self.tokenizer.eot_token
