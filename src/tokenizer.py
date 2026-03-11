from abc import ABC, abstractmethod
from typing import List
import tiktoken

class Tokenizer(ABC):
    """
    Abstract base class for tokenizers.
    """
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encodes a given text into a list of token IDs.
        Args:
            text (str): The input text to be tokenized.
        Returns:
            List[int]: A list of token IDs.
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        Args:
            tokens (List[int]): The list of token IDs to be decoded.
        Returns:
            str: The decoded string.
        """
        pass

    @abstractmethod
    def get_end_token(self) -> int:
        """
        Returns the token ID that represents the end of a sequence.
        Returns:
            int: The end token ID.
        """
        pass

class TiktokenTokenizer(Tokenizer):
    """
    Tokenizer implementation as wrapper of the tiktoken library.
    """
    def __init__(self,
                 encoding_name: str = "cl100k_base"):
        """
        Initializes the TiktokenTokenizer with a specified encoding.
        Args:
            encoding_name (str): The name of the tiktoken encoding to use.
        """
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        """
        Encodes a given text into a list of token IDs.
        Args:
            text (str): The input text to be tokenized.
        Returns:
            List[int]: A list of token IDs.
        """
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        Args:
            tokens (List[int]): The list of token IDs to be decoded.
        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(tokens)
    
    def get_end_token(self) -> int:
        """
        Returns the token ID that represents the end of a sequence.
        Returns:
            int: The end token ID.
        """
        return self.tokenizer.eot_token