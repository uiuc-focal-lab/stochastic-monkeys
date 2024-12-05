from augmentation.augmentation import Augmentation

class NoAugmentation(Augmentation):
    """An augmentation that does nothing (i.e. identity function)."""

    def __call__(
        self,
        prompt: str,
    ) -> str:
        """Returns the prompt unchanged.

        Args:
            prompt: The prompt to augment.
        """

        return prompt