from abc import ABC, abstractmethod
import string

ASCII_SAMPLE_SPACE = string.digits + string.ascii_letters + \
    string.punctuation + " "

class Augmentation(ABC):
    """Abstract class for an input augmentation."""

    @abstractmethod
    def __call__(
        self,
        prompt: str,
    ) -> str:
        """Augments the given prompt.

        Args:
            prompt: The prompt to augment.
        """

        pass