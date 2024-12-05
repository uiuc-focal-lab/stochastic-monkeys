import importlib
import json

AUGMENTATION_NOT_FOUND = (
    "Augmentation \"{name}\" not found. Available augmentations are: "
    "{valid_options}"
)
KEY_FORMATTER = "\"{key}\""
KEY_LIST_SEPARATOR = ", "
AUGMENTATION_MAP_PATH = "augmentation/augmentation_map.json"

_augmentation_map = json.load(
    open(AUGMENTATION_MAP_PATH, "r"),
)

def get_augmentation_class(
    augmentation_name: str,
) -> type:
    """Returns the class for the given augmentation name.
    
    Args:
        augmentation_name: The name of the augmentation.
    """
    if augmentation_name not in _augmentation_map:
        raise ValueError(
            AUGMENTATION_NOT_FOUND.format(
                name=augmentation_name,
                valid_options=KEY_LIST_SEPARATOR.join(
                    [
                        KEY_FORMATTER.format(
                            key=key,
                        ) for key in _augmentation_map.keys()
                    ]
                ),
            )
        )

    augmentation_info = _augmentation_map[augmentation_name]
    
    module = importlib.import_module(augmentation_info["module"])
    augmentation_class = getattr(
        module,
        augmentation_info["class"],
    )

    return augmentation_class