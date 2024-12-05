import random
import string
from augmentation.augmentation import Augmentation, ASCII_SAMPLE_SPACE

RANDOM_STRING_JOIN_SEPARATOR = ""
PREFIX_AUGMENTED_PROMPT = "{prefix} {prompt}"
SUFFIX_AUGMENTED_PROMPT = "{prompt} {suffix}"

NO_PERCENTAGE_LENGTH_ERROR_MESSAGE = (
    "Either `percentage` or `length` should be specified."
)
PERCENTAGE_LENGTH_ERROR_MESSAGE = (
    "Only one of `percentage` or `length` should be specified."
)

class RandomAnywhere(Augmentation):
    """An augmentation to insert a random string at a random location."""

    def __init__(
        self,
        percentage: float = None,
        length: int = None,
    ):
        """Initializes the augmentation.

        Only one of `percentage` or `length` should be provided. The other 
        should be `None`.

        Args:
            percentage: The percentage of the original prompt length that the 
                string length should be.
            length: The length of the string.
        """

        if percentage is not None and length is not None:
            raise ValueError(PERCENTAGE_LENGTH_ERROR_MESSAGE)
        elif percentage is None and length is None:
            raise ValueError(NO_PERCENTAGE_LENGTH_ERROR_MESSAGE)
        
        self.percentage = percentage
        self.length = length

        self._sample_space = ASCII_SAMPLE_SPACE
    
    def __call__(
        self,
        prompt: str,
    ) -> str:
        """Inserts a random string into the prompt at a random location.

        Args:
            prompt: The prompt to augment.
        """

        if self.length is not None:
            string_length = self.length
        else:
            string_length = int(self.percentage * len(prompt))

        random_insertion = RANDOM_STRING_JOIN_SEPARATOR.join(
            random.choices(
                self._sample_space,
                k=string_length,
            )
        )

        index = random.randint(0, len(prompt))

        prompt = prompt[:index] + random_insertion + prompt[index:]

        return prompt

class RandomPrefix(Augmentation):
    """A random prefix augmentation."""

    def __init__(
        self,
        percentage: float = None,
        length: int = None,
    ):
        """Initializes the augmentation.

        Only one of `percentage` or `length` should be provided. The other
        should be `None`.

        Args:
            percentage: The percentage of the original prompt length that the
                string length should be.
            length: The length of the string.
        """

        if percentage is not None and length is not None:
            raise ValueError(PERCENTAGE_LENGTH_ERROR_MESSAGE)
        elif percentage is None and length is None:
            raise ValueError(NO_PERCENTAGE_LENGTH_ERROR_MESSAGE)

        self.percentage = percentage
        self.length = length

        self._sample_space = ASCII_SAMPLE_SPACE
    
    def __call__(
        self,
        prompt: str,
    ) -> str:
        """Adds a random prefix to the prompt.

        Args:
            prompt: The prompt to augment.
        """

        if self.length is not None:
            string_length = self.length
        else:
            string_length = int(self.percentage * len(prompt))

        random_prefix = RANDOM_STRING_JOIN_SEPARATOR.join(
            random.choices(
                self._sample_space,
                k=string_length,
            )
        )
        
        prompt = PREFIX_AUGMENTED_PROMPT.format(
            prompt=prompt,
            prefix=random_prefix,
        )

        return prompt

class RandomSuffix(Augmentation):
    """A random suffix augmentation."""

    def __init__(
        self,
        percentage: float = None,
        length: int = None,
    ):
        """Initializes the augmentation.

        Only one of `percentage` or `length` should be provided. The other
        should be `None`.

        Args:
            percentage: The percentage of the original prompt length that the
                string length should be.
            length: The length of the string.
        """

        if percentage is not None and length is not None:
            raise ValueError(PERCENTAGE_LENGTH_ERROR_MESSAGE)
        elif percentage is None and length is None:
            raise ValueError(NO_PERCENTAGE_LENGTH_ERROR_MESSAGE)

        self.percentage = percentage
        self.length = length

        self._sample_space = ASCII_SAMPLE_SPACE
    
    def __call__(
        self,
        prompt: str,
    ) -> str:
        """Adds a random suffix to the prompt.

        Args:
            prompt: The prompt to augment.
        """

        if self.length is not None:
            string_length = self.length
        else:
            string_length = int(self.percentage * len(prompt))

        random_suffix = RANDOM_STRING_JOIN_SEPARATOR.join(
            random.choices(
                self._sample_space,
                k=string_length,
            )
        )

        prompt = SUFFIX_AUGMENTED_PROMPT.format(
            prompt=prompt,
            suffix=random_suffix,
        )

        return prompt