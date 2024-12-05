from abc import ABC, abstractmethod
import csv
import json

PROMPT_LINE = "{index}: {prompt}"
PROMPT_LINE_SEPARATOR = "\n"

OPEN_FILE_MODE = "r"
FILE_TYPE_DELIMITER = "."

CSV = "csv"
JSONL = "jsonl"

class Dataset(ABC):
    """Abstract class for a dataset."""

    @abstractmethod
    def __init__(
        self,
        dataset_path: str,
    ):
        """Initializes the dataset.
        
        Args:
            dataset_path: The path to the dataset file, which is expected to be
                a CSV file.
        """

        file_type = dataset_path.split(FILE_TYPE_DELIMITER)[-1]

        with open(dataset_path, OPEN_FILE_MODE) as file:
            if file_type == CSV:
                reader = csv.DictReader(file)

                self.dataset = list(reader)
            elif file_type == JSONL:
                self.dataset = [json.loads(line) for line in file]
        
        self._current_index = 0

    def __iter__(self):
        return self
    
    def __next__(self) -> str:
        """Returns the next prompt in the dataset."""
        
        if self._current_index >= len(self):
            self._current_index = 0

            raise StopIteration
        
        prompt = self[self._current_index]

        self._current_index += 1
        
        return prompt
    
    def __repr__(self) -> str:
        """Returns a string representation of the dataset."""

        representation = ""

        for index, prompt in enumerate(self):
            representation += PROMPT_LINE.format(
                index=index,
                prompt=prompt,
            )

            if index != len(self) - 1:
                representation += PROMPT_LINE_SEPARATOR
        
        return representation
    
    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(
        self,
        index: int,
    ) -> str:
        """Returns the prompt at the given index in the dataset.
        
        Args:
            index: The index of the prompt to return.
        """
        pass