from abc import ABC, abstractmethod

import tiktoken


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode a text string into a list of token IDs."""

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back into a string."""

    @abstractmethod
    def get_end_token(self) -> int:
        """Return the token ID that represents the end of a sequence."""


class TiktokenTokenizer(Tokenizer):
    """Tokenizer implementation wrapping the tiktoken library."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialise the TiktokenTokenizer with a specified encoding."""
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        """Encode a text string into a list of token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        return self.tokenizer.decode(tokens)

    def get_end_token(self) -> int:
        """Return the token ID that represents the end of a sequence."""
        return self.tokenizer.eot_token
