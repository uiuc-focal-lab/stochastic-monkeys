import random
import string
from augmentation.augmentation import Augmentation, ASCII_SAMPLE_SPACE
from nltk.tokenize import word_tokenize

PROMPT_STRING_JOIN_SEPARATOR = ""

class RandomWordDeletion(Augmentation):
    """A random edit augmentation."""

    def __init__(
        self,
        percentage: float,
    ):
        """Initializes the augmentation.

        Args:
            percentage: The percentage of characters to delete.
        """

        self.percentage = percentage
        self._sample_space = ASCII_SAMPLE_SPACE
    
    def __call__(
        self,
        prompt: str,
    ) -> str:
        """Randomly deletes a fixed percentage of words in the prompt."""
        words = word_tokenize(prompt)
        word_index = []
        for i, word in enumerate(words):
            if word not in string.punctuation:
                word_index.append(i)
        delete_indices = random.sample(
            range(len(word_index)),
            k=int(self.percentage * len(word_index)),
        )

        for index in delete_indices:
            words[index] = ''
        
        result = []
        for i, token in enumerate(words):
            if i > 0 and token not in string.punctuation and token != '':
                result.append(' ')
            result.append(token)
        return ''.join(result)