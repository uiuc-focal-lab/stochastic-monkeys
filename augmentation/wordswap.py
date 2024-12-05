import random
import string
from augmentation.augmentation import Augmentation, ASCII_SAMPLE_SPACE
from nltk.tokenize import word_tokenize

PROMPT_STRING_JOIN_SEPARATOR = ""

class RandomWordSwap(Augmentation):
    """A random edit augmentation."""

    def __init__(
        self,
        percentage: float,
    ):
        """Initializes the augmentation.

        Args:
            percentage: The percentage of characters to edit.
        """

        self.percentage = percentage
        self._sample_space = ASCII_SAMPLE_SPACE
    
    def __call__(
        self,
        prompt: str,
    ) -> str:
        """Randomly swap a fixed percentage of words in the prompt."""
        words = word_tokenize(prompt)
        word_index = []
        for i, word in enumerate(words):
            if word not in string.punctuation:
                word_index.append(i)
        swap_indices_1 = random.sample(
            range(len(word_index)),
            k=int(self.percentage * len(word_index)),
        )
        swap_indices_2 = random.sample(
            range(len(word_index)),
            k=int(self.percentage * len(word_index)),
        )
        for index_1,index_2 in zip(swap_indices_1,swap_indices_2):
            tmp = words[index_1]
            words[index_1] = words[index_2]
            words[index_2] = tmp
        result = []
        for i, token in enumerate(words):
            if i > 0 and token not in string.punctuation:
                result.append(' ')
            result.append(token)
        return ''.join(result)