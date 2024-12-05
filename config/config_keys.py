"""This file specifies valid keys for different kinds of configuration files.

Valid keys are split into required and optional keys. Each set of keys is 
specified as a dictionary whose keys are the names of the valid keys expected
to appear in the configuration file and whose values are ConfigAttribute 
instances. 
"""

from dataclasses import dataclass

@dataclass
class ConfigAttribute():
    """A configuration attribute.

    When a configuration dictionary is converted to a Config object, the values
    in the configuration dictionary are assigned to the attributes of the Config
    object. The name of the attribute can be specified to be different from the
    name of the key in the dictionary.

    Attributes:
        attribute_name: The name of the attribute for the Config object.
        value_type: The expected type of the value for the attribute.
        default: The default value for the attribute. Only used with optional
            keys.
    """
    attribute_name: str
    value_type: type
    default: type = None

def get_key_from_attribute_name(
    attribute_name: str,
    config_keys: dict[str, ConfigAttribute],
) -> str:
    """Get the key name from an attribute name in a configuration dictionary.

    Args:
        attribute_name: The name of the attribute for the Config object.
        config_keys: A dictionary of configuration keys.

    Returns:
        The key name in the configuration dictionary that corresponds to the
        attribute name.
    """
    for key, config_attribute in config_keys.items():
        if config_attribute.attribute_name == attribute_name:
            return key
    
    return None

BASE_CONFIG_KEY = {
    "base_config": ConfigAttribute(
        attribute_name="base_config",
        value_type=str,
    ),
}

# Model configuration keys
REQUIRED_MODEL_CONFIG_KEYS = {
    "model_name_or_path": ConfigAttribute(
        attribute_name="name_or_path",
        value_type=str,
    ),
    "api": ConfigAttribute(
        attribute_name="api",
        value_type=str,
    ),
}
OPTIONAL_MODEL_CONFIG_KEYS = {
    **BASE_CONFIG_KEY,
    "gpus": ConfigAttribute(
        attribute_name="gpus",
        value_type=int,
    ),
    "model_kwargs": ConfigAttribute(
        attribute_name="model_kwargs",
        value_type=dict,
        default={},
    ),
    "tokenizer_kwargs": ConfigAttribute(
        attribute_name="tokenizer_kwargs",
        value_type=dict,
        default={},
    ),
    "api_key_location": ConfigAttribute(
        attribute_name="api_key_location",
        value_type=str,
    ),
}

# Judge configuration keys
REQUIRED_JUDGE_CONFIG_KEYS = {
    **REQUIRED_MODEL_CONFIG_KEYS,
    "gpus": ConfigAttribute(
        attribute_name="gpus",
        value_type=int,
    ),
    "apply_chat_template": ConfigAttribute(
        attribute_name="apply_chat_template",
        value_type=bool,
    ),
    "judge_prompt": ConfigAttribute(
        attribute_name="judge_prompt",
        value_type=str,
    ),
    "zero_label": ConfigAttribute(
        attribute_name="zero_label",
        value_type=str,
    ),
    "one_label": ConfigAttribute(
        attribute_name="one_label",
        value_type=str,
    ),
}
OPTIONAL_JUDGE_CONFIG_KEYS = {
    **BASE_CONFIG_KEY,
    "model_kwargs": ConfigAttribute(
        attribute_name="model_kwargs",
        value_type=dict,
        default={},
    ),
    "tokenizer_kwargs": ConfigAttribute(
        attribute_name="tokenizer_kwargs",
        value_type=dict,
        default={},
    ),
}

# Experiment configuration keys
REQUIRED_EXPERIMENT_CONFIG_KEYS = {
    "models": ConfigAttribute(
        attribute_name="models",
        value_type=list,
    ),
    "judges": ConfigAttribute(
        attribute_name="judges",
        value_type=list,
    ),
    "dataset": ConfigAttribute(
        attribute_name="dataset",
        value_type=str,
    ),
    "dataset_path": ConfigAttribute(
        attribute_name="dataset_path",
        value_type=str,
    ),
    "random_seed": ConfigAttribute(
        attribute_name="seed",
        value_type=int,
    ),
    "random_input_augmentations": ConfigAttribute(
        attribute_name="augmentations",
        value_type=list,
    ),
    "max_new_tokens": ConfigAttribute(
        attribute_name="max_new_tokens",
        value_type=int,
    ),
    "judge_max_new_tokens": ConfigAttribute(
        attribute_name="judge_max_new_tokens",
        value_type=int,
    ),
    "greedy_generation": ConfigAttribute(
        attribute_name="greedy",
        value_type=bool,
    ),
}
OPTIONAL_EXPERIMENT_CONFIG_KEYS = {
    **BASE_CONFIG_KEY,
    "temperature": ConfigAttribute(
        attribute_name="temperature",
        value_type=float,
        default=1.0,
    ),
    "top_p": ConfigAttribute(
        attribute_name="top_p",
        value_type=float,
        default=1.0,
    ),
    "huggingface_generation_kwargs": ConfigAttribute(
        attribute_name="hf_generation_kwargs",
        value_type=dict,
        default={},
    ),
    "vllm_sampling_kwargs": ConfigAttribute(
        attribute_name="vllm_sampling_kwargs",
        value_type=dict,
        default={},
    ),
    "vllm_use_tqdm": ConfigAttribute(
        attribute_name="vllm_use_tqdm",
        value_type=bool,
        default=False,
    ),
    "split_ray_generation_tasks_by_augmentation": ConfigAttribute(
        attribute_name="split_generation_tasks_by_augmentation",
        value_type=bool,
        default=True,
    ),
    "comparisons": ConfigAttribute(
        attribute_name="comparisons",
        value_type=list,
        default=[],
    ),
    "system_prompt": ConfigAttribute(
        attribute_name="system_prompt",
        value_type=str,
        default="",
    ),
    "output_samples": ConfigAttribute(
        attribute_name="output_samples",
        value_type=int,
        default=1,
    ),
    "pair_attack_model": ConfigAttribute(
        attribute_name="pair_attack_model",
        value_type=str,
        default="",
    ),
    "pair_attack_judge_model": ConfigAttribute(
        attribute_name="pair_attack_judge_model",
        value_type=str,
        default="",
    ),
    "pair_breadth": ConfigAttribute(
        attribute_name="pair_breadth",
        value_type=int,
        default=9,
    ),
    "pair_depth": ConfigAttribute(
        attribute_name="pair_depth",
        value_type=int,
        default=3,
    ),
    "pair_attack_model_greedy_generation": ConfigAttribute(
        attribute_name="pair_attack_model_greedy_generation",
        value_type=bool,
        default=False,
    ),
    "pair_attack_model_temperature": ConfigAttribute(
        attribute_name="pair_attack_model_temperature",
        value_type=float,
        default=1.0,
    ),
    "pair_attack_model_top_p": ConfigAttribute(
        attribute_name="pair_attack_model_top_p",
        value_type=float,
        default=1.0,
    ),
}

# Augmentation configuration keys
REQUIRED_AUGMENTATION_CONFIG_KEYS = {
    "name": ConfigAttribute(
        attribute_name="name",
        value_type=str,
    ),
}

OPTIONAL_AUGMENTATION_CONFIG_KEYS = {
    "augmentation_samples": ConfigAttribute(
        attribute_name="augmentation_samples",
        value_type=int,
        default=1,
    ),
    "thresholds": ConfigAttribute(
        attribute_name="thresholds",
        value_type=list,
        default=[0],
    ),
    "kwargs": ConfigAttribute(
        attribute_name="kwargs",
        value_type=dict,
        default={},
    ),
}

# Comparisons configuration keys
REQUIRED_COMPARISONS_CONFIG_KEYS = {
    "models": ConfigAttribute(
        attribute_name="models",
        value_type=list,
    ),
    "augmentations": ConfigAttribute(
        attribute_name="augmentations",
        value_type=list,
    ),
    "other_results": ConfigAttribute(
        attribute_name="other_results",
        value_type=list,
    ),
}

OPTIONAL_COMPARISONS_CONFIG_KEYS = {
    "thresholds": ConfigAttribute(
        attribute_name="thresholds",
        value_type=list,
        default=[0],
    ),
}

REQUIRED_COMPARISONS_OTHER_RESULTS_CONFIG_KEYS = {
    "experiment_name": ConfigAttribute(
        attribute_name="experiment_name",
        value_type=str,
    ),
    "models": ConfigAttribute(
        attribute_name="models",
        value_type=list,
    ),
    "models_comparison": ConfigAttribute(
        attribute_name="models_comparison",
        value_type=str,
    ),
    "augmentations": ConfigAttribute(
        attribute_name="augmentations",
        value_type=list,
    ),
    "augmentations_comparison": ConfigAttribute(
        attribute_name="augmentations_comparison",
        value_type=str,
    ),
}

OPTIONAL_COMPARISONS_OTHER_RESULTS_CONFIG_KEYS = {
    "thresholds": ConfigAttribute(
        attribute_name="thresholds",
        value_type=list,
        default=[0],
    ),
    "thresholds_comparison": ConfigAttribute(
        attribute_name="thresholds_comparison",
        value_type=str,
        default="elementwise",
    ),
}