import importlib
import json

DATASET_NOT_FOUND = (
    "Dataset \"{name}\" not found. Available datasets are: {valid_options}"
)
KEY_FORMATTER = "\"{key}\""
KEY_LIST_SEPARATOR = ", "
DATASET_MAP_PATH = "dataset/dataset_map.json"

_dataset_map = json.load(
    open(DATASET_MAP_PATH, "r"),
)

def get_dataset_class(
    dataset_name: str,
) -> type:
    """Returns the class for the given dataset name.
    
    Args:
        dataset_name: The name of the dataset.
    """
    if dataset_name not in _dataset_map:
        raise ValueError(
            DATASET_NOT_FOUND.format(
                name=dataset_name,
                valid_options=KEY_LIST_SEPARATOR.join(
                    [
                        KEY_FORMATTER.format(
                            key=key,
                        ) for key in _dataset_map.keys()
                    ]
                ),
            )
        )

    dataset_info = _dataset_map[dataset_name]
    
    module = importlib.import_module(dataset_info["module"])
    dataset_class = getattr(
        module,
        dataset_info["class"],
    )

    return dataset_class