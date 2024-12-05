import tomllib
import warnings
import json
from config.config_keys import (
    get_key_from_attribute_name,
    ConfigAttribute,
    BASE_CONFIG_KEY,
    REQUIRED_MODEL_CONFIG_KEYS,
    OPTIONAL_MODEL_CONFIG_KEYS,
    REQUIRED_JUDGE_CONFIG_KEYS,
    OPTIONAL_JUDGE_CONFIG_KEYS,
    REQUIRED_EXPERIMENT_CONFIG_KEYS,
    OPTIONAL_EXPERIMENT_CONFIG_KEYS,
    REQUIRED_AUGMENTATION_CONFIG_KEYS,
    OPTIONAL_AUGMENTATION_CONFIG_KEYS,
    REQUIRED_COMPARISONS_CONFIG_KEYS,
    OPTIONAL_COMPARISONS_CONFIG_KEYS,
    REQUIRED_COMPARISONS_OTHER_RESULTS_CONFIG_KEYS,
    OPTIONAL_COMPARISONS_OTHER_RESULTS_CONFIG_KEYS,
)

CONFIG_NAME_NOT_FOUND_ERROR = (
    "Configuration for \"{name}\" not found in \"{config_path}\". Valid options are"
    ": {valid_options}"
)
CONFIG_MISSING_REQUIRED_KEYS_ERROR = (
    "Required key(s) not found in the configuration for \"{name}\": {keys}"
)
CONFIG_UNRECOGNIZED_KEYS_WARNING = (
    "Ignoring unrecognized key(s) found in the configuration for \"{name}\": "
    "{keys}"
)
CONFIG_IMPROPER_TYPE_ERROR_BASE = (
    "At least one key has a value with the wrong type in the configuration for"
    " \"{name}\":"
)
CONFIG_IMPROPER_TYPE_ERROR_INDIVIDUAL = (
    " \"{key}\": value is of type {value_type}, but expected type "
    "{expected_type}"
)
CONFIG_IMPROPER_TYPE_ERROR_INDIVIDUAL_SEPARATOR = ";"
CONFIG_PRINT_INDENT = 4
KEY_LIST_SEPARATOR = ", "
KEY_FORMATTER = "\"{key}\""
LOAD_CONFIG_OPEN_MODE = "rb"
BASE_CONFIG_KEY_NAME = list(BASE_CONFIG_KEY)[0]
OTHER_RESULTS_KEY = get_key_from_attribute_name(
    attribute_name="other_results",
    config_keys=REQUIRED_COMPARISONS_CONFIG_KEYS,
)
LOAD_COMPARISONS_OTHER_RESULTS_CONFIG_NAME = (
    "{base_config_name}."
    f"{OTHER_RESULTS_KEY}"
    "[{i}]"
)

def validate_config_required_keys(
    config: dict,
    name: str,
    required_keys: dict[str, ConfigAttribute],
):
    """Validates that the configuration contains all required keys.

    Does nothing if the configuration contains all required keys. Otherwise,
    raises an error with a list of the missing keys.

    Args:
        config: Model configuration dictionary.
        name: The name associated with the configuration
        required_keys: A dictionary where the keys are the names of required
            keys in the configuration.
    
    Raises:
        KeyError: A required key is missing from the configuration.
    """

    missing_required_keys = [
        key for key in required_keys if key not in config
    ]

    if len(missing_required_keys) > 0:
        if len(missing_required_keys) == 1:
            keys = KEY_FORMATTER.format(
                key=missing_required_keys[0],
            )
        else:
            keys = KEY_LIST_SEPARATOR.join(
                [
                    KEY_FORMATTER.format(
                        key=key
                    ) for key in missing_required_keys
                ]
            )

        raise KeyError(
            CONFIG_MISSING_REQUIRED_KEYS_ERROR.format(
                name=name,
                keys=keys,
            )
        )

def validate_config_value_types(
    config: dict,
    name: str,
    valid_keys: dict[str, ConfigAttribute],
):
    """Validates the types of the values in the configuration.

    Does nothing if all values have the correct types. Otherwise, raises an 
    error with a list of the keys that have values of the wrong type.

    Args:
        config: Model configuration dictionary.
        name: The name associated with the configuration.
        valid_keys: A dictionary where the keys are the names of valid keys 
            that may appear in the configuration.
    
    Raises:
        TypeError: At least one key has a value with the wrong type.
    """

    improper_keys = []

    for key, value in config.items():
        if key in valid_keys:
            expected_type = valid_keys[key].value_type

            if not isinstance(value, expected_type):
                improper_keys.append((key, expected_type, type(value)))

    if len(improper_keys) > 0:
        error_message = CONFIG_IMPROPER_TYPE_ERROR_BASE.format(
            name=name,
        )

        for i, (key, expected_type, value_type) in enumerate(improper_keys):
            error_message += CONFIG_IMPROPER_TYPE_ERROR_INDIVIDUAL.format(
                key=key,
                value_type=value_type.__name__,
                expected_type=expected_type.__name__,
            )

            if i < len(improper_keys) - 1:
                error_message += CONFIG_IMPROPER_TYPE_ERROR_INDIVIDUAL_SEPARATOR

        raise TypeError(error_message)

def check_config_unrecognized_keys(
    config: dict,
    name: str,
    valid_keys: dict[str, ConfigAttribute],
):
    """
    Check for unrecognized keys in the configuration.

    Does nothing if all keys are recognized. Otherwise, raises a warning with a
    list of the unrecognized keys.

    Args:
        config: Model configuration dictionary.
        name: The name associated with the configuration.
        valid_keys: A dictionary where the keys are the names of valid keys 
            that may appear in the configuration.
    """

    unrecognized_keys = [
        key for key in config if key not in valid_keys
    ]

    if len(unrecognized_keys) > 0:
        if len(unrecognized_keys) == 1:
            keys = KEY_FORMATTER.format(
                key=unrecognized_keys[0],
            )
        else:
            keys = KEY_LIST_SEPARATOR.join(
                [
                    KEY_FORMATTER.format(
                        key=key,
                    ) for key in unrecognized_keys
                ]
            )

        warnings.warn(
            CONFIG_UNRECOGNIZED_KEYS_WARNING.format(
                name=name,
                keys=keys,
            )
        )

def validate_config(
    config: dict,
    name: str,
    required_keys: dict[str, ConfigAttribute],
    optional_keys: dict[str, ConfigAttribute],
):
    """Validates the configuration.

    Validates that the configuration contains all required keys, that all keys
    have the correct types, and checks whether there are any unrecognized keys.

    Args:
        config: Model configuration dictionary.
        name: The name associated with the configuration.
        required_keys: A dictionary where the keys are the names of required
            keys in the configuration.
        optional_keys: A dictionary where the keys are the names of optional
            keys that may appear in the configuration.
    """
    all_keys = required_keys | optional_keys

    validate_config_required_keys(
        config=config,
        name=name,
        required_keys=required_keys,
    )
    validate_config_value_types(
        config=config,
        name=name,
        valid_keys=all_keys,
    )
    check_config_unrecognized_keys(
        config=config,
        name=name,
        valid_keys=all_keys,
    )

class ConfigJSONEncoder(json.JSONEncoder):
    """A JSON encoder that adds support for Config objects."""
    
    def default(self, o):
        if isinstance(o, Config):
            return o.__dict__
        
        return super().default(o)

class Config():
    """A configuration object.

    This provides an interface between the names of keys used in a configuration
    dictionary and the names of the attributes used in the code.
    """

    def __init__(
        self,
        config: dict,
        name: str,
        required_keys: dict[str, ConfigAttribute],
        optional_keys: dict[str, ConfigAttribute],
    ) -> None:
        """Initializes the configuration.
        
        Sets the class attributes to the values in the configuration dictionary.
        Unrecognized keys are ignored. The configuration is validated before 
        the attributes are set.

        Args:
            config: Configuration dictionary.
            name: The name associated with the configuration.
            required_keys: A dictionary where the keys are the names of required
                keys in the configuration.
            optional_keys: A dictionary where the keys are the names of optional
                keys that may appear in the configuration.
        """

        validate_config(
            config=config,
            name=name,
            required_keys=required_keys,
            optional_keys=optional_keys,
        )

        all_keys = required_keys | optional_keys
        
        for key in all_keys:
            setattr(
                self,
                all_keys[key].attribute_name,
                config.get(
                    key,
                    all_keys[key].default,
                )
            )
    
    def __repr__(self) -> str:
        pretty_dict = json.dumps(
            self.__dict__,
            sort_keys=True,
            indent=CONFIG_PRINT_INDENT,
            cls=ConfigJSONEncoder,
        )

        return pretty_dict

def load_config(
    name: str,
    config_path: str,
    required_keys: dict[str, ConfigAttribute],
    optional_keys: dict[str, ConfigAttribute],
) -> Config:
    """Returns a configuration from a configuration file.

    Assumes the configuration file is in TOML format. Each first level table
    in the configuration file is treated as a separate configuration.

    Args:
        name: The name of the first level table to retrieve in the configuration
            file.
        config_path: Path to the configuration file.
        required_keys: A dictionary where the keys are the names of required
            keys in the configuration.
        optional_keys: A dictionary where the keys are the names of optional
            keys that may appear in the configuration.
        config_type: The type of configuration to return.
    """

    all_configs = tomllib.load(
        open(
            config_path,
            LOAD_CONFIG_OPEN_MODE,
        ),
    )

    def get_config_dict_from_name(
        all_configs: dict,
        name: str,
    ) -> dict:
        """Returns the configuration dictionary associated with the name.

        If the configuration has a base configuration, the base configuration
        is retrieved first and then the current configuration is imposed on top 
        of the base configuration.

        Args:
            all_configs: A dictionary of configuration dictionaries.
            name: The configuration dictionary key.
        
        Raises:
            KeyError: The name is not found in all_configs.
        """
        try:
            config = all_configs[name]
        except KeyError:
            raise ValueError(
                CONFIG_NAME_NOT_FOUND_ERROR.format(
                    name=name,
                    config_path=config_path,
                    valid_options=KEY_LIST_SEPARATOR.join(
                        [
                            KEY_FORMATTER.format(
                                key=key
                            ) for key in list(all_configs.keys())
                        ]
                    ),
                )
            )

        # Get base configuration if exists    
        if BASE_CONFIG_KEY_NAME in config:
            base_config_name = config[BASE_CONFIG_KEY_NAME]

            base_config = get_config_dict_from_name(
                all_configs=all_configs,
                name=base_config_name,
            )

            # Add missing keys from base configuration to current configuration
            for key in base_config:
                if key not in config:
                    config[key] = base_config[key]
            
        return config
        
    config = Config(
        config=get_config_dict_from_name(
            all_configs=all_configs,
            name=name,
        ),
        name=name,
        required_keys=required_keys,
        optional_keys=optional_keys,
    )

    return config

def load_model_config(
    name: str,
    config_path: str,
) -> Config:
    """Returns a model configuration.

    Args:
        name: The first level table to retrieve in the configuration file.
        config_path: Path to the model configuration file.
    """

    return load_config(
        name=name,
        config_path=config_path,
        required_keys=REQUIRED_MODEL_CONFIG_KEYS,
        optional_keys=OPTIONAL_MODEL_CONFIG_KEYS,
    )

def load_judge_config(
    name: str,
    config_path: str,
) -> Config:
    """Returns a judge configuration.

    Args:
        name: The first level table to retrieve in the configuration file.
        config_path: Path to the model configuration file.
    """

    return load_config(
        name=name,
        config_path=config_path,
        required_keys=REQUIRED_JUDGE_CONFIG_KEYS,
        optional_keys=OPTIONAL_JUDGE_CONFIG_KEYS,
    )

def load_experiment_config(
    name: str,
    config_path: str,
) -> Config:
    """Returns an experiment configuration.

    Args:
        name: The first level table to retrieve in the configuration file.
        config_path: Path to the experiment configuration file.
    """

    return load_config(
        name=name,
        config_path=config_path,
        required_keys=REQUIRED_EXPERIMENT_CONFIG_KEYS,
        optional_keys=OPTIONAL_EXPERIMENT_CONFIG_KEYS,
    )

def load_augmentation_config(
    name: str,
    config_dict: dict,
) -> Config:
    """Returns an augmentation configuration.

    Args:
        name: The name associated with the configuration.
        config_dict: The configuration dictionary. Should be retrieved from an
            experiment configuration.
    """

    config = Config(
        config=config_dict,
        name=name,
        required_keys=REQUIRED_AUGMENTATION_CONFIG_KEYS,
        optional_keys=OPTIONAL_AUGMENTATION_CONFIG_KEYS,
    )

    return config

def load_comparisons_config(
    name: str,
    config_dict: dict,
) -> Config:
    """Returns a comparison configuration.

    Args:
        name: The name associated with the configuration.
        config_dict: The configuration dictionary. Should be retrieved from an
            experiment configuration.
    """

    config = Config(
        config=config_dict,
        name=name,
        required_keys=REQUIRED_COMPARISONS_CONFIG_KEYS,
        optional_keys=OPTIONAL_COMPARISONS_CONFIG_KEYS,
    )

    for i, other_results_config in enumerate(config.other_results):
        other_results_config = Config(
            config=other_results_config,
            name=LOAD_COMPARISONS_OTHER_RESULTS_CONFIG_NAME.format(
                base_config_name=name,
                i=i,
            ),
            required_keys=(
                REQUIRED_COMPARISONS_OTHER_RESULTS_CONFIG_KEYS
            ),
            optional_keys=(
                OPTIONAL_COMPARISONS_OTHER_RESULTS_CONFIG_KEYS
            ),
        )

        config.other_results[i] = other_results_config

    return config