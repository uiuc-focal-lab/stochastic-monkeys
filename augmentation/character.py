import random
import string
from augmentation.augmentation import Augmentation, ASCII_SAMPLE_SPACE

PROMPT_STRING_JOIN_SEPARATOR = ""

class RandomEdit(Augmentation):
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
        """Randomly edits a fixed percentage of characters in the prompt."""

        edit_indices = random.sample(
            range(len(prompt)),
            k=int(self.percentage * len(prompt)),
        )

        prompt = list(prompt)

        for index in edit_indices:
            prompt[index] = random.choice(self._sample_space)
        
        prompt = PROMPT_STRING_JOIN_SEPARATOR.join(prompt)

        return prompt

class RandomInsertion(Augmentation):
    """A random insertion augmentation."""

    def __init__(
        self,
        percentage: float,
    ):
        """Initializes the augmentation.

        Args:
            percentage: The percentage of the original prompt length that the
                number of characters to insert should be.
        """

        self.percentage = percentage
        self._sample_space = ASCII_SAMPLE_SPACE
    
    def __call__(
        self,
        prompt: str,
    ) -> str:
        """Inserts random characters into the prompt at random locations.

        Args:
            prompt: The prompt to augment.
        """

        random_insertion = PROMPT_STRING_JOIN_SEPARATOR.join(
            random.choices(
                self._sample_space,
                k=int(len(prompt) * self.percentage),
            )
        )

        indices = random.sample(
            range(len(prompt)),
            k=len(random_insertion),
        )

        for i, index in enumerate(sorted(indices)):
            prompt = \
                prompt[:index + i] + random_insertion[i] + prompt[index + i:]
        
        return prompt

class RandomDeletion(Augmentation):
    """A random character deletion augmentation."""

    def __init__(
        self,
        percentage: float,
    ):
        """Initializes the augmentation.

        Args:
            percentage: The percentage of characters to delete.
        """

        self.percentage = percentage
    
    def __call__(
        self,
        prompt: str,
    ) -> str:
        """Randomly edits a fixed percentage of characters in the prompt."""

        delete_indices = random.sample(
            range(len(prompt)),
            k=int(self.percentage * len(prompt)),
        )

        prompt = [
            prompt[i] for i in range(len(prompt)) if i not in delete_indices
        ]
        
        prompt = PROMPT_STRING_JOIN_SEPARATOR.join(prompt)

        return prompt