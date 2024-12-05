import random
import string
from augmentation.augmentation import Augmentation, ASCII_SAMPLE_SPACE

RANDOM_STRING_JOIN_SEPARATOR = ""
keyboard_mapping = {
    'q': ['1','2','w','s','a'],
    'w': ['q', '2', '3', 'e', 'd', 's', 'a'],
    'e': ['w', '3', '4', 'r', 'f', 'd' ,'s'],
    'r': ['e', '4', '5', 't', 'g', 'f', 'd'],
    't': ['r', '5', '6', 'y', 'h', 'g', 'f'],
    'y': ['g', 't', '6', '7', 'u', 'j', 'h'],
    'u': ['y', '7', '8', 'i', 'k', 'j', 'h'],
    'i': ['j', 'u', '8', '9', 'o', 'l', 'k'],
    'o': ['i', '9', '0', 'p', ';', 'l', 'k'],
    'p': ['o', '0', '-', '[', '\'', ';', 'l'],
    'a': ['q', 'w', 's', 'x', 'z'],
    's': ['a', 'w', 'e', 'd', 'c', 'x', 'z'],
    'd': ['s', 'e', 'r', 'f', 'v', 'c', 'x'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'i', 'o', 'l', '.', ',', 'm'],
    'l': ['k', 'o', 'p', ';', '.', ','],
    'z': ['a', 's', 'x'],
    'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'],
    'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k', ',']
}

class RandomTypo(Augmentation):
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
    
    def __call__(
        self,
        prompt: str,
    ) -> str:
        """Randomly edits a fixed percentage of characters in the prompt."""
        char_index =  [i for i, char in enumerate(prompt) if char.isalpha()]
        edit_indices = random.sample(
            char_index,
            k=int(self.percentage * len(char_index)),
        )

        prompt = list(prompt)

        for index in edit_indices:
            character = prompt[index].lower()
            typo = random.choice(keyboard_mapping[character])
            prompt[index] = typo
        
        prompt = RANDOM_STRING_JOIN_SEPARATOR.join(prompt)

        return prompt