from dataset.dataset import Dataset

class SorryBench(Dataset):
    """Dataset class for the SORRY-Bench dataset."""
    
    def __init__(
        self,
        dataset_path: str,
    ):
        """Initializes the dataset.
        
        Args:
            dataset_path: The path to the dataset file, which is expected to be
                a CSV file.
        """
        super().__init__(dataset_path)

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self.dataset)
    
    def __getitem__(
        self,
        index: int
    ) -> str:
        """Returns the prompt at the given index in the dataset.
        
        Args:
            index: The index of the prompt to return.
        """
        return self.dataset[index]["prompt"]
    
    def get_target(
        self,
        index: int
    ) -> str:
        """Returns the target string at the given index in the dataset.
        
        Args:
            index: The index of the target string to return.
        """
        return self.dataset[index]["target"]