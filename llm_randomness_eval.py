import random
import argparse
import logging
from pathlib import Path
import sys
import json
import os
import itertools
from enum import StrEnum, auto
import time
import torch
import ray
from openai import OpenAI
from config.config_utils import (
    Config,
    load_model_config,
    load_judge_config,
    load_experiment_config,
    load_augmentation_config,
    load_comparisons_config,
)
from config.config_keys import (
    get_key_from_attribute_name,
    REQUIRED_EXPERIMENT_CONFIG_KEYS,
    OPTIONAL_EXPERIMENT_CONFIG_KEYS,
)
from model_utils import (
    load_model,
    load_tokenizer,
    get_generation_inputs,
    get_generation_outputs,
    create_openai_input_file,
    API,
    INVALID_API_ERROR,
)
from dataset import get_dataset_class
from dataset.dataset import Dataset
from augmentation import get_augmentation_class
from augmentation.augmentation import Augmentation
from pair import (
    PAIR_SYSTEM_PROMPT,
    PAIR_INIT_PROMPT,
    PAIR_PROCESS_RESPONSE,
    PAIR_INIT_ASSISTANT_PREFILL,
    PAIR_ASSISTANT_PREFILL,
)

EXPERIMENT_MESSAGE_PREFIX = "(\"{experiment_name}\") "
EXPERIMENT_AUGMENTATION_MESSAGE_PREFIX = (
    "(\"{experiment_name}\":\"{augmentation_name}\") "
)
EXPERIMENT_MODEL_MESSAGE_PREFIX = "(\"{experiment_name}\":\"{model_alias}\") "
ALL_MESSAGE_PREFIX = (
    "(\"{experiment_name}\":\"{model_alias}\":\"{augmentation_name}\") "
)

SKIPPING_MESSAGE = "Skipping..."
SKIPPING_EXPERIMENT_MESSAGE = "Skipping experiment..."
SKIPPING_CALCULATION_MESSAGE = "Skipping calculation..."
SKIPPING_COMPARISONS_MESSAGE = "Skipping comparisons for it..."

DUPLICATE_EXPERIMENT_MESSAGE = (
    "Experiment \"{experiment_name}\" already started. "
    f"{SKIPPING_MESSAGE}"
)
DUPLICATE_AUGMENTATION_MESSAGE = (
    f"{EXPERIMENT_MESSAGE_PREFIX}"
    "Augmentation \"{augmentation_name}\" already loaded. "
    f"{SKIPPING_MESSAGE}"
)
DUPLICATE_MODEL_MESSAGE = (
    f"{EXPERIMENT_MESSAGE_PREFIX}"
    "Model \"{model_alias}\" already started. "
    f"{SKIPPING_MESSAGE}"
)
PREVIOUS_RESULTS_FOUND_MESSAGE = (
    f"{ALL_MESSAGE_PREFIX}"
    "Loaded previous results found at \"{results_path}\"."
)

AUGMENTATION_UNIQUE_NAME_KEY_VALUE_PAIR = "_{key}_{value}"
AUGMENTATION_UNIQUE_NAME_SUFFIX = "_samples_{samples}"
AUGMENTATION_UNIQUE_NAME_VALUE_REPLACEMENTS = [
    (".", "_"),
]

EXPERIMENT_AUGMENTATIONS_KEY = get_key_from_attribute_name(
    attribute_name="augmentations",
    config_keys=REQUIRED_EXPERIMENT_CONFIG_KEYS,
)
COMPARISONS_KEY = get_key_from_attribute_name(
    attribute_name="comparisons",
    config_keys=OPTIONAL_EXPERIMENT_CONFIG_KEYS,
)

LOAD_AUGMENTATION_CONFIG_NAME = (
    "{experiment_name}."
    f"{EXPERIMENT_AUGMENTATIONS_KEY}"
    "[{i}]"
)
COMPARISONS_NAME = (
    "{experiment_name}."
    f"{COMPARISONS_KEY}"
    "[{i}]"
)

EXPERIMENT_CONFIG_MESSAGE = (
    f"{EXPERIMENT_MESSAGE_PREFIX}"
    "Loaded experiment configuration:\n{config}"
)
JUDGE_CONFIG_MESSAGE = (
    f"{EXPERIMENT_MESSAGE_PREFIX}"
    "Loaded \"{judge_alias}\" judge configuration:\n{config}"
)
JUDGE_CONFIG_PREVIOUSLY_LOADED_MESSAGE = (
    f"{EXPERIMENT_MESSAGE_PREFIX}"
    "Judge configuration for \"{judge_alias}\" previously loaded."
)
MODEL_CONFIG_MESSAGE = (
    f"{EXPERIMENT_MODEL_MESSAGE_PREFIX}"
    "Loaded model configuration:\n{config}"
)
MODEL_CONFIG_PREVIOUSLY_LOADED_MESSAGE = (
    f"{EXPERIMENT_MODEL_MESSAGE_PREFIX}"
    "Model configuration previously loaded."
)

AUGMENTATION_KWARGS_FOR_UNIQUE_NAME = " ({augmentation_kwargs})"
AUGMENTATION_MESSAGE_SAMPLES = " {aug_samples} samples of"
AUGMENTATION_MESSAGE = (
    f"{EXPERIMENT_AUGMENTATION_MESSAGE_PREFIX}"
    "Augmented each prompt with{augmentation_samples} the "
    "\"{augmentation_base_name}\"{augmentation_kwargs} augmentation."
)
AUGMENT_PROMPT_ERROR = (
    f"{EXPERIMENT_AUGMENTATION_MESSAGE_PREFIX}"
    "Error augmenting prompts:\n\t{message}\n"
    f"{SKIPPING_MESSAGE}"
)
SAMPLES_ERROR_MESSAGE = (
    f"{EXPERIMENT_MESSAGE_PREFIX}"
    "At least one of augmentation_samples and output_samples must be 1. Got: "
    "augmentation_samples={augs_samples}, "
    "output_samples={output_samples}."
)

STARTED_GENERATION_TASK_MESSAGE = (
    f"{EXPERIMENT_MODEL_MESSAGE_PREFIX}Started generation task."
)
STARTED_GENERATION_TASK_SPLIT_MESSAGE = (
    f"{ALL_MESSAGE_PREFIX}Started generation task."
)
COMPLETED_GENERATION_TASK_MESSAGE = (
    f"{EXPERIMENT_MODEL_MESSAGE_PREFIX}Completed generation task. "
    "Started evaluation task..."
)
COMPLETED_GENERATION_TASK_SPLIT_MESSAGE = (
    f"{ALL_MESSAGE_PREFIX}Completed generation task. Started evaluation task..."
)
COMPLETED_EVALUATION_MESSAGE = (
    f"{ALL_MESSAGE_PREFIX}Completed evaluation task."
)
SUCCESS_RATE_MESSAGE = (
    f"{ALL_MESSAGE_PREFIX}Success rate "
    "(judge={judge_alias}, threshold="
    "{threshold:.4f}): {success_rate:.4f}"
)
SAVED_RESULTS_MESSAGE = (
    f"{ALL_MESSAGE_PREFIX}Saved results to "
    "\"{save_path}\"."
)
COMPLETED_EVALUATIONS_MESSAGE = "Completed evaluations."

AVERAGE_INPUT_LENGTH = (
    f"{ALL_MESSAGE_PREFIX}"
    "Average input length: {average_length_chars:.2f} characters, "
    "{average_length_tokens:.2f} tokens"
)

ERROR_MESSAGE = "{message}"

LOAD_EXPERIMENT_CONFIG_ERROR = (
    "Error loading experiment configuration for \"{name}\" from "
    "\"{config_path}\":\n\t{message}\n"
    f"{SKIPPING_MESSAGE}"
)
LOAD_JUDGE_ERROR = (
    f"{EXPERIMENT_MESSAGE_PREFIX}"
    "Error loading judge configuration:\n\t{message}\n"
    f"{SKIPPING_EXPERIMENT_MESSAGE}"
)
LOAD_DATASET_ERROR = (
    f"{EXPERIMENT_MESSAGE_PREFIX}"
    "Error loading dataset:\n\t{message}\n"
    f"{SKIPPING_EXPERIMENT_MESSAGE}"
)
LOAD_AUGMENTATION_ERROR = (
    f"{EXPERIMENT_MESSAGE_PREFIX}"
    "Error loading augmentation configuration:\n\t{message}\n"
    f"{SKIPPING_MESSAGE}"
)
LOAD_MODEL_ERROR = (
    f"{EXPERIMENT_MESSAGE_PREFIX}"
    "Error loading model configuration:\n\t{message}\n"
    f"{SKIPPING_MESSAGE}"
)

OPENAI_FILE_RESPONSE_DELIMETER = "\n"
OPENAI_REQUEST_ERROR = "OpenAI API request {outcome}."
OPENAI_REQUEST_FAILED = f"{EXPERIMENT_MODEL_MESSAGE_PREFIX}" + \
    OPENAI_REQUEST_ERROR.format(
        outcome="failed"
    )
OPENAI_REQUEST_EXPIRED = f"{EXPERIMENT_MODEL_MESSAGE_PREFIX}" + \
    OPENAI_REQUEST_ERROR.format(
        outcome="expired"
    )
OPENAI_REQUEST_CANCELLED = f"{EXPERIMENT_MODEL_MESSAGE_PREFIX}" + \
    OPENAI_REQUEST_ERROR.format(
        outcome="cancelled"
    )

PAIR_ATTACK_EXPERIMENTS_ERROR = (
    "Only one experiment can be specified for the PAIR attack. Received "
    "{num_experiments} experiments."
)

RESULTS_FILE_INDENT = 4
RESULTS_FILENAME = "{augmentation_name}.json"
SAVE_RESULTS_FILE_OPEN_MODE = "w"
LOOKUP_RESULTS_FILE_OPEN_MODE = "r"
DEFAULT_DEVICE = "cuda"

WARNINGS_LOGGER = "py.warnings"
LOGGER_FORMAT = (
    "[%(levelname)s:%(filename)s:%(lineno)d] %(message)s"
)
WARNINGS_LOGGER_FORMAT = LOGGER_FORMAT
WARNINGS_FORMATTER_SPLIT_DELIMETER = ":"
WARNINGS_FORMATTER_SPLIT_MAXSPLIT = 3
WARNINGS_FORMATTER_FILENAME_SPLIT_DELIMETER = "/"
WARNINGS_FORMATTER_MESSAGE_SPLIT_DELIMETER = "warnings.warn("

class ComparisonType(StrEnum):
    """An enumeration of the types of comparisons that can be made."""
    ELEMENTWISE = auto()
    CARTESIAN = auto()

class AttackType(StrEnum):
    """An enumeration of the types of attacks that can be performed."""
    STOCHASTIC_MONKEYS = auto()
    PAIR = auto()

class WarningsFormatter(logging.Formatter):
    """A custom formatter for warnings."""

    def format(
        self,
        record: logging.LogRecord,
    ):
        """Overrides the format method to apply custom formatting.

        The filename, line number, and message are extracted from the record's
        message in order to be used in the formatted message.

        Args:
            record: The log record to format.
        """

        record_message_split = record.getMessage().split(
            WARNINGS_FORMATTER_SPLIT_DELIMETER,
            WARNINGS_FORMATTER_SPLIT_MAXSPLIT,
        )
        message_split = record_message_split[3].split(
            WARNINGS_FORMATTER_MESSAGE_SPLIT_DELIMETER,
        )

        filename = record_message_split[0]
        lineno = int(record_message_split[1])
        message = "".join(
            message_split[:-1]
        ).strip()

        new_record = logging.LogRecord(
            name=record.name,
            level=record.levelno,
            pathname=filename,
            lineno=lineno,
            msg=message,
            args=record.args,
            exc_info=record.exc_info,
        )

        return super().format(new_record)

def get_args() -> argparse.Namespace:
    """Returns the command line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiments",
        help="Names of the experiments to run",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--experiment_config_path",
        help="Path to the experiment configuration file",
        type=str,
        default="./config/experiments.toml",
    )
    parser.add_argument(
        "--judge_config_path",
        help="Path to the judge configuration file",
        type=str,
        default="./config/judges.toml",
    )
    parser.add_argument(
        "--model_config_path",
        help="Path to the model configuration file",
        type=str,
        default="./config/models.toml",
    )
    parser.add_argument(
        "--results_dir",
        help="Directory to save evaluation results",
        type=str,
        default="./results",
    )
    parser.add_argument(
        "--log_path",
        help="Path to the log file",
        type=str,
        default="./output.log",
    )
    parser.add_argument(
        "--attack",
        help="The type of attack to perform.",
        type=str,
        default=AttackType.STOCHASTIC_MONKEYS.value,
        choices=[
            attack.value for attack in AttackType
        ],
    )

    args = parser.parse_args()

    return args

def get_logger(
    log_path: str,
) -> logging.Logger:
    """Returns a logger for logging.

    Args:
        log_path: The path to the log file.
    """
    logging.captureWarnings(True)

    # Standard logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger_formatter = logging.Formatter(LOGGER_FORMAT)

    logger_file_handler = logging.FileHandler(
        log_path
    )
    logger_file_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_file_handler)

    logger_stream_handler = logging.StreamHandler(sys.stdout)
    logger_stream_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_stream_handler)

    # Warnings logger
    warnings_logger = logging.getLogger(WARNINGS_LOGGER)

    warnings_logger_formatter = WarningsFormatter(WARNINGS_LOGGER_FORMAT)
    warnings_logger_file_handler = logging.FileHandler(
        log_path
    )
    warnings_logger_file_handler.setFormatter(warnings_logger_formatter)
    warnings_logger.addHandler(warnings_logger_file_handler)

    warnings_logger_stream_handler = logging.StreamHandler(sys.stdout)
    warnings_logger_stream_handler.setFormatter(warnings_logger_formatter)
    warnings_logger.addHandler(warnings_logger_stream_handler)

    return logger

@ray.remote
def generate_outputs(
    system_prompts: list,
    prompts: list,
    prompts_split: list,
    output_samples: int,
    greedy: bool,
    temperature: float,
    top_p: float,
    model_alias: str,
    model_config: Config,
    judge: bool,
    experiment_name: str,
    experiment_config: dict,
    chat_history: list = None,
    prefill: str = "",
    judge_alias: str = None,
    compute_average_prompt_length: bool = True,
) -> dict:
    """Generates outputs from a model for the given prompts.

    Args:
        system_prompts: A list of system prompts to use for each prompt.
        prompts: A list of the latest user prompts to generate outputs for.
        prompts_split: A list of dictionaries describing how prompts is split.
            Each dictionary contains the following keys:
                - start: The starting index of the split.
                - augmentation_name: The name of the augmentation.
                - num_prompts: The number of base prompts in the split.
                - augs_per_prompt: The number of augmentations per prompt.
        output_samples: The number of output samples to generate per prompt.
        greedy: Whether to use greedy decoding.
        temperature: The temperature to use for sampling.
        top_p: The top-p value to use for sampling.
        prefill: A prefill string for the assistant response.
        model_alias: The alias for the model being evaluated.
        model_config: The configuration for the model to load.
        judge: Whether the model is a judge model.
        experiment_name: The name of the experiment.
        experiment_config: The configuration for the experiment.
        chat_history: The chat history so far preceding the user prompts 
            specified in the prompts argument.
        judge_alias: The alias for the judge model, if the model is a judge.
        compute_average_prompt_length: Whether to compute the average prompt
            length.
    
    Returns:
        A dictionary containing the following keys:
            - experiment_name: The name of the experiment.
            - model_alias: The alias of the model.
            - prompts_split: Passed from the prompts_split argument.
            - outputs: The generated outputs.
    """

    model = load_model(
        name_or_path=model_config.name_or_path,
        api=model_config.api,
        **(model_config.model_kwargs),
    )

    tokenizer = load_tokenizer(
        model_name_or_path=model_config.name_or_path,
        **(model_config.tokenizer_kwargs),
    )

    apply_chat_template = model_config.apply_chat_template if judge else True

    if chat_history is None:
        chat_history = [[] for _ in range(len(prompts))]

    inputs = get_generation_inputs(
        system_prompts=system_prompts,
        chat_history=chat_history,
        prompts=prompts,
        prefill=prefill,
        tokenizer=tokenizer,
        api=model_config.api,
        apply_chat_template=apply_chat_template,
    )

    average_prompt_lengths = {}

    if not judge and compute_average_prompt_length:
        # Compute average prompt lengths
        for split in prompts_split:
            start = split["start"]
            augmentation_name = split["augmentation_name"]
            num_prompts = split["num_prompts"]
            augs_per_prompt = split["augs_per_prompt"]

            input_length_chars = []
            input_length_tokens = []

            average_input_length_chars = None
            average_input_length_tokens = None

            for i in range(num_prompts):
                inputs_start = start + i * augs_per_prompt
                inputs_end = inputs_start + augs_per_prompt
                chars = prompts[inputs_start:inputs_end]
                tokens = inputs[inputs_start:inputs_end]

                input_length_chars.extend([len(prompt) for prompt in chars])
                input_length_tokens.extend([len(prompt) for prompt in tokens])
            
            if len(input_length_chars) > 0:
                average_input_length_chars = sum(input_length_chars) / \
                    len(input_length_chars)
            
            if len(input_length_tokens) > 0:
                average_input_length_tokens = sum(input_length_tokens) / \
                    len(input_length_tokens)
        
            average_prompt_lengths[augmentation_name] = {
                "chars": average_input_length_chars,
                "tokens": average_input_length_tokens,
            }
    
    max_new_tokens = experiment_config.judge_max_new_tokens if judge else \
        experiment_config.max_new_tokens

    outputs = get_generation_outputs(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        output_samples=output_samples,
        max_new_tokens=max_new_tokens,
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        random_seed=experiment_config.seed,
        vllm_use_tqdm=experiment_config.vllm_use_tqdm,
    )

    results = {
        "loaded": False,
        "experiment_name": experiment_name,
        "model_alias": model_alias,
        "prompts_split": prompts_split,
        "prompts": prompts,
        "outputs": outputs,
    }

    if judge:
        results["judge_alias"] = judge_alias
    elif compute_average_prompt_length:
        results["average_prompt_lengths"] = average_prompt_lengths

    return results

@ray.remote
def generate_outputs_openai(
    prompts: list,
    prompts_split: list,
    model_alias: str,
    model_config: Config,
    experiment_name: str,
    experiment_config: dict,
    results_dir: str,
) -> dict:
    """Generates outputs from an OpenAI model for the given prompts.

    Args:
        prompts: A list of prompts to generate outputs for.
        prompts_split: A list of dictionaries describing how prompts is split.
            Each dictionary contains the following keys:
                - start: The starting index of the split.
                - augmentation_name: The name of the augmentation.
                - num_prompts: The number of base prompts in the split.
                - augs_per_prompt: The number of augmentations per prompt.
        model_alias: The alias for the model being evaluated.
        model_config: The configuration for the model to load.
        experiment_name: The name of the experiment.
        experiment_config: The configuration for the experiment.
        results_dir: The directory to save results to. Used to save the input 
            file.
        logger: The logger to use for logging.
    
    Returns:
        A dictionary containing the following keys:
            - experiment_name: The name of the experiment.
            - model_alias: The alias of the model.
            - prompts_split: Passed from the prompts_split argument.
            - outputs: The generated outputs.
    """

    client = OpenAI(
        api_key=os.environ[model_config.api_key_location]
    )

    input_file_name = create_openai_input_file(
        model_name=model_config.name_or_path,
        prompts=prompts,
        results_dir=results_dir,
        experiment_name=experiment_name,
        experiment_config=experiment_config,
        model_alias=model_alias,
    )

    # Upload input file
    batch_input_file = client.files.create(
        file=open(input_file_name, "rb"),
        purpose="batch",
    )

    # Submit request for outputs
    request = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": \
                f"llm-randomness-eval[{experiment_name}:{model_alias}]",
        }
    )

    status = ""

    # Wait for request to complete
    while status != "completed":
        status = client.batches.retrieve(request.id).status

        if status == "failed":
            raise Exception(
                OPENAI_REQUEST_FAILED.format(
                    experiment_name=experiment_name,
                    model_alias=model_alias,
                )
            )
        elif status == "expired":
            raise Exception(
                OPENAI_REQUEST_EXPIRED.format(
                    experiment_name=experiment_name,
                    model_alias=model_alias,
                )
            )
        elif status == "cancelled":
            raise Exception(
                OPENAI_REQUEST_CANCELLED.format(
                    experiment_name=experiment_name,
                    model_alias=model_alias,
                )
            )

        time.sleep(60)
    
    # Retrieve results
    file_response = client.files.content(
        client.batches.retrieve(request.id).output_file_id
    )

    # Parse results
    response_split = file_response.text.split(
        OPENAI_FILE_RESPONSE_DELIMETER
    )[:-1]
    outputs = [None] * len(prompts)

    for line in response_split:
        prompt_id = int(json.loads(line)["custom_id"])

        line_results = json.loads(line).get(
            "response",
            {},
        ).get(
            "body",
            {},
        ).get(
            "choices",
            [],
        )[0].get(
            "message",
            {},
        ).get(
            "content",
            "",
        )

        outputs[prompt_id] = [line_results]

    results = {
        "loaded": False,
        "experiment_name": experiment_name,
        "model_alias": model_alias,
        "prompts_split": prompts_split,
        "prompts": prompts,
        "outputs": outputs,
    }

    return results

def get_score(
    output: str,
    zero_label: str,
    one_label: str,
) -> int | None:
    """Returns a score extracted from a judgment model output.

    If the output does not match the zero or one label, None is returned.

    Args:
        output: The output to extract the score from.
        zero_label: The label for a score of 0.
        one_label: The label for a score of 1.
    
    Returns:
        The extracted score, or None if the output does not match the zero or 
        one label.
    """

    output = output.strip().lower()
    score = None

    if output[:len(zero_label)] == zero_label.lower():
        score = 0
    elif output[:len(one_label)] == one_label.lower():
        score = 1
    
    return score

def get_augmentation_unique_name(
    augmentation_config: Config,
    output_samples: int,
) -> str:
    """Returns a unique ID for an augmentation.

    Returns an ID that combines the augmentation name and its keyword arguments.

    Args:
        augmentation_config: The augmentation configuration.
        output_samples: The number of output samples per prompt.
    """

    def dict_to_str(
        d: dict,
    ) -> str:
        """Returns a string representation of a dictionary.
        
        Args:
            d: The dictionary to convert to a string.
        """
        
        str_representation = ""

        for key, value in d.items():
            if value is None:
                continue

            if isinstance(value, dict) and len(value) > 0:
                value = dict_to_str(value)
            else:
                value = str(value)
            
            str_representation += \
                AUGMENTATION_UNIQUE_NAME_KEY_VALUE_PAIR.format(
                    key=key,
                    value=value,
                )
        
        return str_representation

    unique_key = \
        augmentation_config.name + \
        dict_to_str(
            augmentation_config.kwargs,
        ) + \
        AUGMENTATION_UNIQUE_NAME_SUFFIX.format(
            samples=str(
                max(
                    augmentation_config.augmentation_samples,
                    output_samples,
                )
            ),
        )

    for old, new in AUGMENTATION_UNIQUE_NAME_VALUE_REPLACEMENTS:
        unique_key = unique_key.replace(old, new)
    
    return unique_key

def safe_load_comparisons_config(
    config_dict: dict,
    config_idx: int,
    experiment_name: str,
    logger: logging.Logger,
) -> Config | None:
    """Safely loads a comparison configuration.

    If an error occurs when loading the configuration, the error message is
    logged and None is returned.

    Args:
        config_dict: The dictionary containing the configuration.
        config_idx: The index of the configuration in the experiment 
            configuration.
        experiment_name: The name of the current experiment.
        logger: The logger to use for logging.
    
    Returns:
        The loaded comparison configuration, or None if an error occurred.
    """

    comparison_config = None

    try:
        comparison_config = load_comparisons_config(
            name=COMPARISONS_NAME.format(
                experiment_name=experiment_name,
                i=config_idx,
            ),
            config_dict=config_dict,
        )
    except Exception as message:
        logger.error(
            EXPERIMENT_MESSAGE_PREFIX.format(
                experiment_name=experiment_name,
            ) +
            ERROR_MESSAGE.format(
                message=message,
            )
        )
    
    return comparison_config

def safe_load_experiment_config(
    experiment_name: str,
    config_path: str,
    logger: logging.Logger,
    log_config: bool = False,
) -> Config | None:
    """Safely loads an experiment configuration.

    If an error occurs when loading the configuration, the error message is 
    logged and None is returned.

    Args:
        name: The name of the experiment.
        config_path: The path to the experiment configuration file.
        logger: The logger to use for logging.
        log_config: Whether to log the configuration after loading.
    
    Returns:
        The loaded experiment configuration, or None if an error occurred.
    """

    experiment_config = None

    try:
        experiment_config = load_experiment_config(
            name=experiment_name,
            config_path=config_path,
        )
    except Exception as message:
        logger.error(
            LOAD_EXPERIMENT_CONFIG_ERROR.format(
                name=experiment_name,
                config_path=config_path,
                message=message,
            )
        )
    
    if log_config and experiment_config is not None:
        # Output experiment configuration
        logger.info(
            EXPERIMENT_CONFIG_MESSAGE.format(
                experiment_name=experiment_name,
                config=str(experiment_config),
            )
        )
    
    return experiment_config

def safe_load_judge_config(
    judge_alias: str,
    config_path: str,
    cache: dict,
    experiment_name: str,
    logger: logging.Logger,
    log_config: bool = False,
) -> Config | None:
    """Safely loads a judge configuration.

    If an error occurs when loading the configuration, the error message is
    logged and None is returned. If the judge name is already in the cache, the
    cached configuration is returned. Otherwise, the configuration is loaded
    and cached.

    Args:
        judge_alias: The alias of the judge model.
        config_path: The path to the judge configuration file.
        cache: A dictionary mapping judge names to their cached configurations.
        experiment_name: The name of the current experiment.
        logger: The logger to use for logging.
        log_config: Whether to log the configuration after loading.
    
    Returns:
        The loaded judge configuration, or None if an error occurred.
    """

    judge_config = None

    try:
        if judge_alias in cache:
            # Use cached config if available
            judge_config = cache[judge_alias]

            if log_config:
                logger.info(
                    JUDGE_CONFIG_PREVIOUSLY_LOADED_MESSAGE.format(
                        experiment_name=experiment_name,
                        judge_alias=judge_alias,
                    )
                )
        else:
            judge_config = load_judge_config(
                name=judge_alias,
                config_path=config_path,
            )

            # Cache judge config
            cache[judge_alias] = judge_config

            if log_config:
                # Output judge configuration
                logger.info(
                    JUDGE_CONFIG_MESSAGE.format(
                        experiment_name=experiment_name,
                        judge_alias=judge_alias,
                        config=str(judge_config),
                    )
                )
    except Exception as message:
        logger.error(
            LOAD_JUDGE_ERROR.format(
                experiment_name=experiment_name,
                message=message,
            )
        )
    
    return judge_config

def safe_load_dataset(
    dataset_name: str,
    dataset_path: str,
    experiment_name: str,
    logger: logging.Logger,
) -> Dataset | None:
    """Safely loads a dataset.

    If an error occurs when loading the dataset, the error message is logged and
    None is returned.

    Args:
        dataset_name: The name of the dataset.
        dataset_path: The path to the dataset.
        experiment_name: The name of the current experiment.
        logger: The logger to use for logging.
    
    Returns:
        The loaded dataset, or None if an error occurred.
    """

    dataset = None

    try:
        dataset_class = get_dataset_class(
            dataset_name=dataset_name,
        )
        dataset = dataset_class(
            dataset_path=dataset_path,
        )
    except Exception as message:
        logger.error(
            LOAD_DATASET_ERROR.format(
                experiment_name=experiment_name,
                message=message,
            )
        )
    
    return dataset

def safe_load_augmentation(
    config_dict: dict,
    augmentation_index: int,
    experiment_name: str,
    logger: logging.Logger,
) -> tuple[Augmentation, Config] | None:
    """Safely loads an augmentation configuration.

    If an error occurs when loading the configuration, the error message is 
    logged and None is returned.

    Args:
        config_dict: The dictionary containing the configuration.
        augmentation_index: The index of the augmentation in the experiment 
            configuration.
        experiment_name: The name of the current experiment.
        logger: The logger to use for logging.
    
    Returns:
        The loaded augmentation and its configuration, or None if an error 
        occurred.
    """

    loaded_augmentation = None
    load_config_name = LOAD_AUGMENTATION_CONFIG_NAME.format(
        experiment_name=experiment_name,
        i=augmentation_index,
    )

    try:
        augmentation_config = load_augmentation_config(
            name=load_config_name,
            config_dict=config_dict,
        )
        augmentation_class = get_augmentation_class(
            augmentation_name=augmentation_config.name,
        )
        augmentation = augmentation_class(
            **augmentation_config.kwargs,
        )

        loaded_augmentation = (augmentation, augmentation_config)
    except Exception as message:
        logger.error(
            LOAD_AUGMENTATION_ERROR.format(
                experiment_name=experiment_name,
                message=message,
            )
        )
    
    return loaded_augmentation

def safe_load_model_config(
    model_alias: str,
    config_path: str,
    cache: dict,
    experiment_name: str,
    logger: logging.Logger,
    log_config: bool = False,
    return_error: bool = False,
) -> Config | None | Exception:
    """Safely loads a model configuration.

    If an error occurs when loading the configuration, the error message is 
    logged and None is returned OR the error message is directly returned,
    depending on the value of return_error.

    Args:
        model_alias: The alias of the model.
        config_path: The path to the model configuration file.
        cache: A dictionary mapping model aliases to their cached 
            configurations.
        experiment_name: The name of the current experiment.
        logger: The logger to use for logging.
        log_config: Whether to log the configuration after loading.
        return_error: Whether to return the error message if an error occurs. If
            False, the error message is logged and None is returned.
    
    Returns:
        The loaded model configuration, or None if an error occurred.
    """

    model_config = None

    try:
        if model_alias in cache:
            model_config = cache[model_alias]

            if log_config:
                logger.info(
                    MODEL_CONFIG_PREVIOUSLY_LOADED_MESSAGE.format(
                        experiment_name=experiment_name,
                        model_alias=model_alias,
                    )
                )
        else:
            model_config = load_model_config(
                name=model_alias,
                config_path=config_path,
            )

            cache[model_alias] = model_config

            if log_config:
                # Output model configuration
                logger.info(
                    MODEL_CONFIG_MESSAGE.format(
                        experiment_name=experiment_name,
                        model_alias=model_alias,
                        config=str(model_config),
                    )
                )
    except Exception as message:
        if return_error:
            model_config = message
        else:
            logger.error(
                LOAD_MODEL_ERROR.format(
                    experiment_name=experiment_name,
                    message=message,
                )
            )
    
    return model_config

def safe_augment_prompts(
    dataset: Dataset,
    augmentation: Augmentation,
    augmentation_name: str,
    augmentation_config: Config,
    random_seed: int,
    experiment_name: str,
    logger: logging.Logger,
    log_augmentation: bool = False,
) -> list:
    """Safely augments a list of prompts.

    Augments the prompts in the dataset with the given augmentation. If an error
    occurs when augmenting the prompts, the error message is logged and None is
    returned.

    Args:
        dataset: The dataset to augment.
        augmentation: The augmentation to apply to the prompts.
        augmentation_name: The name of the augmentation.
        augmentation_config: The configuration for the augmentation.
        random_seed: The random seed to set before augmenting the prompts.
        experiment_name: The name of the current experiment.
        logger: The logger to use for logging.
        log_augmentation: Whether to log a message upon successfully
            augmenting the prompts.
    
    Returns:
        The augmented prompts, or an empty list if an error occurred.
    """

    random.seed(augmentation_name + str(random_seed))

    augmented_prompts = None

    try:
        augmented_prompts = [
            [
                augmentation(prompt) for _ in range(
                    augmentation_config.augmentation_samples,
                )
            ] for prompt in dataset
        ]
    except Exception as message:
        logger.error(
            AUGMENT_PROMPT_ERROR.format(
                experiment_name=experiment_name,
                augmentation_name=augmentation_name,
                message=message,
            )
        )

    if log_augmentation:
        if len(augmentation_config.kwargs) > 0:
            augmentation_kwargs_str = \
                AUGMENTATION_KWARGS_FOR_UNIQUE_NAME.format(
                    augmentation_kwargs=str(
                        augmentation_config.kwargs
                    )[1:-1],
                )
        else:
            augmentation_kwargs_str = ""

        logger.info(
            AUGMENTATION_MESSAGE.format(
                experiment_name=experiment_name,
                augmentation_base_name=augmentation_config.name,
                augmentation_samples=AUGMENTATION_MESSAGE_SAMPLES.format(
                    aug_samples=augmentation_config.augmentation_samples,
                ) if augmentation_config.augmentation_samples > 1 else "",
                augmentation_kwargs=augmentation_kwargs_str,
                augmentation_name=augmentation_name,
            )
        )
    
    return augmented_prompts

@ray.remote
def loaded_generation_data_task(
    experiment_name: str,
    model_alias: str,
    prompts_split: list,
    prompts: list,
    outputs: list,
):
    """A task that immediately returns loaded generation results.

    Args:
        experiment_name: The name of the experiment.
        model_alias: The alias of the model.
        prompts_split: A list of dictionaries describing how prompts is split.
            Each dictionary contains the following keys:
                - start: The starting index of the split.
                - augmentation_name: The name of the augmentation.
                - num_prompts: The number of base prompts in the split.
                - augs_per_prompt: The number of augmentations per prompt.
        prompts: The prompts used to generate the outputs.
        outputs: The loaded outputs.
    """

    results = {
        "loaded": True,
        "experiment_name": experiment_name,
        "model_alias": model_alias,
        "prompts_split": prompts_split,
        "prompts": prompts,
        "outputs": outputs,
    }

    return results

def queue_model_generation_tasks(
    experiment_name: str,
    model_alias: str,
    augmented_prompts: dict,
    attack_targets: dict,
    dataset: Dataset,
    results_dir: str,
    experiment_config: Config,
    model_config: Config,
    attacker_model_config: Config,
    attacker_judge_config: Config,
    judge_configs: dict,
    outputs_ids: list,
    task_queue: list,
    added_experiments: list,
    experiment_added_models: list,
    results_data: dict,
    attack: str,
    logger: logging.Logger,
):
    """Queues up generation tasks to be started for a model.

    Queues up generation tasks for each augmentation in the experiment for the
    given model. If split_generation_tasks_by_augmentation is true in the
    experiment configuration, a task is queued up for each augmentation.
    Otherwise, a single task is queued up for all augmentations.

    Args:
        experiment_name: The name of the experiment.
        model_alias: The alias of the model.
        augmented_prompts: A dictionary mapping augmentation names to a list of 
            lists. Each inner list corresponds to a base prompt and contains 
            the same number of augmentation samples for that prompt.
        attack_targets: A dictionary mapping augmentation names to a list of
            lists. Each inner list corresponds to a prompt and contains targets
            for each augmentation sample for that prompt.
        dataset: The original dataset used.
        results_dir: The directory to look up existing evaluation results.
        experiment_config: The configuration for the experiment.
        model_config: The configuration for the model.
        attacker_model_config: The configuration for the attacker model, if 
            applicable.
        attacker_judge_config: The configuration for the attacker judge, if
            applicable.
        judge_configs: A dictionary mapping experiment names to a dictionary
            mapping judge aliases to their configurations.
        outputs_ids: A list of output IDs for the tasks queued up.
        task_queue: The list to queue up tasks for generation. Each task is 
            represented by a dictionary containing the following keys:
                - task_arguments: A dictionary containing the arguments for the
                    generate_outputs function.
                - task_launch_message: The message to log when the task is 
                    launched.
        added_experiments: A list of experiments that already have at least one
            task added to the task queue.
        experiment_added_models: A list of model aliases that already have at 
            least one task added to the task queue for the current experiment.
        results_data: A dictionary mapping experiment names to a 
            dictionary mapping model aliases to a dictionary mapping 
            augmentation names to a dictionary containing the results.
        attack: The type of attack to perform.
        logger: The logger to use for logging.
    """

    generate_all_prompts = []
    generate_prompts_split = []
    loaded_all_prompts = []
    evaluate_all_outputs = []
    evaluate_outputs_split = []

    all_attack_targets = []

    expected_num_scores = len(judge_configs[experiment_name])

    # Loop through each augmentation
    for augmentation_name, prompts in augmented_prompts.items():
        generate_augmentation = False
        evaluate_augmentation = False

        loaded_prompts = []
        evaluate_outputs = []

        results_path = Path(
            results_dir,
            experiment_name,
            model_alias,
            RESULTS_FILENAME.format(
                augmentation_name=augmentation_name,
            ),
        )
        results = None
        results_exist = (
            results_path.exists() and 
            results_path.is_file()
        )

        if experiment_name not in added_experiments:
            added_experiments.append(experiment_name)

        if model_alias not in experiment_added_models:
            experiment_added_models.append(model_alias)

        if results_exist:
            # If previous results exist, load them into results_data
            with results_path.open(LOOKUP_RESULTS_FILE_OPEN_MODE) as f:
                results = json.load(f)

                loaded_prompts = list(itertools.chain.from_iterable([
                    [
                        augmentation["prompt"] 
                            for augmentation in result["augmentations"]
                    ] for result in results["results"]
                ]))

                num_scores = len(
                    results["results"][0]["augmentations"][0].get(
                        "score",
                        {},
                    )
                )

                # Check if evaluation has not been completed yet
                if num_scores < expected_num_scores:
                    evaluate_augmentation = True

                    if experiment_config.output_samples > 1:
                        evaluate_outputs = [
                            [
                                augmentation["output"] 
                                    for augmentation in result["augmentations"]
                            ] for result in results["results"]
                        ]
                    else:
                        evaluate_outputs = [
                            [augmentation["output"]]
                                for result in results["results"]
                                for augmentation in result["augmentations"]
                        ]
                
                # Reset comparison results, if they exist
                if "success_rate_gain" in results["summary"]:
                    results["summary"].pop("success_rate_gain")
                
                if "average_score_gain" in results["summary"]:
                    results["summary"].pop("average_score_gain")

                results_data.setdefault(
                    experiment_name,
                    {},
                ).setdefault(
                    model_alias,
                    {},
                )[augmentation_name] = results
            
            logger.info(
                PREVIOUS_RESULTS_FOUND_MESSAGE.format(
                    experiment_name=experiment_name,
                    model_alias=model_alias,
                    augmentation_name=augmentation_name,
                    results_path=results_path,
                )
            )
        else:
            generate_augmentation = True

        if generate_augmentation:
            split = {
                "start": len(generate_all_prompts),
                "augmentation_name": augmentation_name,
                "num_prompts": len(prompts),
                "augs_per_prompt": len(prompts[0]),
            }

            flattened_prompts = list(itertools.chain(*prompts))

            if attack == AttackType.PAIR.value:
                flattened_attack_targets = \
                    list(itertools.chain(*attack_targets[augmentation_name]))

            if experiment_config.split_generation_tasks_by_augmentation:
                # Queue up task arguments for each augmentation
                task_arguments = {
                    "prompts": flattened_prompts,
                    "prompts_split": [split],
                    "model_alias": model_alias,
                    "model_config": model_config,
                    "experiment_name": experiment_name,
                    "experiment_config": experiment_config,
                }

                if attack == AttackType.STOCHASTIC_MONKEYS.value:
                    task_arguments["system_prompts"] = [
                        experiment_config.system_prompt \
                            for _ in flattened_prompts
                    ]
                    task_arguments["output_samples"] = \
                        experiment_config.output_samples
                    task_arguments["judge"] = False
                    task_arguments["greedy"] = experiment_config.greedy
                    task_arguments["temperature"] = \
                        experiment_config.temperature
                    task_arguments["top_p"] = experiment_config.top_p
                elif attack == AttackType.PAIR.value:
                    task_arguments["attacker_model_config"] = \
                        attacker_model_config
                    task_arguments["attacker_judge_config"] = \
                        attacker_judge_config
                    task_arguments["attack_targets"] = flattened_attack_targets
                    task_arguments["dataset"] = dataset

                task_launch_message = \
                    STARTED_GENERATION_TASK_SPLIT_MESSAGE.format(
                        experiment_name=experiment_name,
                        model_alias=model_alias,
                        augmentation_name=augmentation_name,
                    )
                
                task_queue.append({
                    "task_arguments": task_arguments,
                    "task_launch_message": task_launch_message,
                })
            else:
                generate_all_prompts.extend(flattened_prompts)
                generate_prompts_split.append(split)

                if attack == AttackType.PAIR.value:
                    all_attack_targets.extend(flattened_attack_targets)
        elif evaluate_augmentation:
            if attack == AttackType.PAIR.value:
                augs_per_prompt = len(
                    results["results"][0]["augmentations"]
                )
            else:
                augs_per_prompt = len(prompts[0])

            split = {
                "start": len(evaluate_all_outputs),
                "augmentation_name": augmentation_name,
                "num_prompts": len(prompts),
                "augs_per_prompt": augs_per_prompt,
            }

            if experiment_config.split_generation_tasks_by_augmentation:
                # evaluate_all_outputs is always empty, so "start" is always 0

                outputs_ids.append(
                    loaded_generation_data_task.remote(
                        experiment_name=experiment_name,
                        model_alias=model_alias,
                        prompts_split=[split],
                        prompts=loaded_prompts,
                        outputs=evaluate_outputs,
                    )
                )
            else:
                loaded_all_prompts.extend(loaded_prompts)
                evaluate_all_outputs.extend(evaluate_outputs)
                evaluate_outputs_split.append(split)
    
    if not experiment_config.split_generation_tasks_by_augmentation:
        if len(generate_all_prompts) > 0:
            # Queue up task arguments for all augmentations
            task_arguments = {
                "prompts": generate_all_prompts,
                "prompts_split": generate_prompts_split,
                "model_alias": model_alias,
                "model_config": model_config,
                "experiment_name": experiment_name,
                "experiment_config": experiment_config,
            }

            if attack == AttackType.STOCHASTIC_MONKEYS.value:
                task_arguments["system_prompts"] = [
                    experiment_config.system_prompt \
                        for _ in generate_all_prompts
                ]
                task_arguments["output_samples"] = \
                    experiment_config.output_samples
                task_arguments["judge"] = False
                task_arguments["greedy"] = experiment_config.greedy
                task_arguments["temperature"] = experiment_config.temperature
                task_arguments["top_p"] = experiment_config.top_p
            elif attack == AttackType.PAIR.value:
                task_arguments["attacker_model_config"] = attacker_model_config
                task_arguments["attacker_judge_config"] = attacker_judge_config
                task_arguments["attack_targets"] = all_attack_targets
                task_arguments["dataset"] = dataset

            task_launch_message = \
                STARTED_GENERATION_TASK_MESSAGE.format(
                    experiment_name=experiment_name,
                    model_alias=model_alias,
                )
            
            task_queue.append({
                "task_arguments": task_arguments,
                "task_launch_message": task_launch_message,
            })
        
        if len(evaluate_all_outputs) > 0:
            outputs_ids.append(
                loaded_generation_data_task.remote(
                    experiment_name=experiment_name,
                    model_alias=model_alias,
                    prompts_split=evaluate_outputs_split,
                    prompts=loaded_all_prompts,
                    outputs=evaluate_all_outputs,
                )
            )

@ray.remote
def pair_attack(
    prompts: list,
    prompts_split: list,
    attack_targets: list,
    dataset: Dataset,
    model_alias: str,
    model_config: Config,
    attacker_model_config: Config,
    attacker_judge_config: Config,
    experiment_name: str,
    experiment_config: dict,
):
    """Perform the PAIR attack for the given prompts.

    At the end of the attack, augmented_prompts is updated for the current
    experiment to include the jailbreak prompts and outputs. prompts_split is
    also adjusted accordingly.

    Args:
        prompts: A list of prompts to generate outputs for.
        prompts_split: A list of dictionaries describing how prompts is split.
            Each dictionary contains the following keys:
                - start: The starting index of the split.
                - augmentation_name: The name of the augmentation.
                - num_prompts: The number of base prompts in the split.
                - augs_per_prompt: The number of augmentations per prompt.
        augmented_prompts: The augemented prompts for the current experiment.
        attack_targets: A list of targets for each prompt.
        dataset: The original dataset used.
        model_alias: The alias for the target model being evaluated.
        model_config: The configuration for the target model.
        attacker_model_config: The configuration for the attacker model.
        attacker_judge_config: The configuration for the attacker judge model.
        experiment_name: The name of the experiment.
        experiment_config: The configuration for the experiment.
    
    Returns:
        A dictionary containing the following keys:
            - experiment_name: The name of the experiment.
            - model_alias: The alias of the model.
            - prompts_split: Passed from the prompts_split argument.
            - outputs: The generated outputs.
    """

    jailbreak_candidates = [[] for _ in prompts]
    target_outputs = [[] for _ in prompts]

    system_prompts = [
        PAIR_SYSTEM_PROMPT.format(
            goal=prompt,
            target_str=target,
        ) for (prompt, target) in zip(prompts, attack_targets)
    ]
    latest_prompts = [
        PAIR_INIT_PROMPT.format(
            goal=prompt,
            target=target,
        ) for (prompt, target) in zip(prompts, attack_targets)
    ]
    chat_history = [[] for _ in prompts]
    
    for i in range(experiment_config.pair_depth):
        # Step 1: Query attacker model for new candidate jailbreak prompts
        prefill = PAIR_INIT_ASSISTANT_PREFILL if i == 0 else \
            PAIR_ASSISTANT_PREFILL
                 
        new_candidates_results = ray.get(
            generate_outputs.options(
                num_gpus=attacker_model_config.gpus,
            ).remote(
                system_prompts=system_prompts,
                prompts=latest_prompts,
                prompts_split=prompts_split,
                output_samples=(
                    experiment_config.pair_breadth if i == 0 else 1
                ),
                greedy=experiment_config.pair_attack_model_greedy_generation,
                temperature=experiment_config.pair_attack_model_temperature,
                top_p=experiment_config.pair_attack_model_top_p,
                model_alias="",
                model_config=attacker_model_config,
                judge=False,
                experiment_name=experiment_name,
                experiment_config=experiment_config,
                chat_history=chat_history,
                prefill=prefill,
                compute_average_prompt_length=False,
            )
        )

        new_candidates = []

        for history, prompt, outputs in zip(
            chat_history,
            latest_prompts,
            new_candidates_results["outputs"]
        ):
            for output in outputs:
                # Extract candidate jailbreak prompt from output
                json_output = prefill.strip() + output
                open_bracket = json_output.find("{")
                close_bracket = json_output.find("}") + 1

                try:
                    extracted_candidate = json_output.split("\"prompt\": \"")[1]
                    extracted_candidate = extracted_candidate.split("}")[0]
                    final_quote = extracted_candidate.rfind("\"")
                    extracted_candidate = extracted_candidate[:final_quote]
                except:
                    extracted_candidate = output
                
                new_candidates.append(extracted_candidate)

                # Update chat history
                if i == 0:
                    history.append([
                        prompt,
                        json_output,
                    ])
                else:
                    history.extend([
                        prompt,
                        prefill.strip() + outputs[0],
                    ])
        
        if i == 0:
            chat_history = list(itertools.chain.from_iterable(chat_history))
            system_prompts = list(itertools.chain.from_iterable([
                [system_prompt] * experiment_config.pair_breadth for \
                    system_prompt in system_prompts
            ]))

            """Note: prompts_split will be incorrect after this point, but it
            is not used in the rest of the attack. We update prompts_split at
            the end just before returning."""
        
        # Record candidates to return at the end
        for j in range(len(prompts)):
            for k in range(experiment_config.pair_breadth):
                location = j * experiment_config.pair_breadth + k
                jailbreak_candidates[j].append(
                    new_candidates[location]
                )

        # Step 2: Apply candidate jailbreak prompt to target model
        target_results = ray.get(
            generate_outputs.options(
                num_gpus=model_config.gpus,
            ).remote(
                system_prompts=[
                    experiment_config.system_prompt for _ in new_candidates
                ],
                prompts=new_candidates,
                prompts_split=prompts_split,
                output_samples=1,
                greedy=experiment_config.greedy,
                temperature=experiment_config.temperature,
                top_p=experiment_config.top_p,
                model_alias=model_alias,
                model_config=model_config,
                judge=False,
                experiment_name=experiment_name,
                experiment_config=experiment_config,
                compute_average_prompt_length=False,
            )
        )

        judge_prompts = [[] for _ in new_candidates]

        # Record target outputs to return at the end, and create judge prompts
        for j in range(len(prompts)):
            for k in range(experiment_config.pair_breadth):
                location = j * experiment_config.pair_breadth + k
                target_outputs[j].append(
                    target_results["outputs"][location]
                )

                if i < experiment_config.pair_depth - 1:
                    judge_prompts[j].append(
                        attacker_judge_config.judge_prompt.format(
                            user_prompt=dataset[j],
                            model_output=target_results["outputs"][location],
                        )
                    )
        
        if i == experiment_config.pair_depth - 1:
            break
        
        judge_prompts = list(itertools.chain.from_iterable(judge_prompts))
    
        # Step 3: Query attack judge model for success scores
        judge_results = ray.get(
            generate_outputs.options(
                num_gpus=attacker_judge_config.gpus,
            ).remote(
                system_prompts=[
                    "" for _ in prompts
                ],
                prompts=judge_prompts,
                prompts_split=prompts_split,
                output_samples=1,
                greedy=True,
                temperature=0,
                top_p=1,
                model_alias=model_alias,
                model_config=attacker_judge_config,
                judge=True,
                experiment_name=experiment_name,
                experiment_config=experiment_config,
                judge_alias="",
            )
        )

        judge_scores = [
            get_score(
                output=output[0],
                zero_label=attacker_judge_config.zero_label,
                one_label=attacker_judge_config.one_label,
            ) for output in judge_results["outputs"]
        ]

        latest_prompts = [
            PAIR_PROCESS_RESPONSE.format(
                target_response=target_output[0],
                goal=prompt,
                score=score,
            ) for target_output, prompt, score in zip(
                target_results["outputs"],
                new_candidates,
                judge_scores,
            )
        ]
    
    jailbreak_candidates = list(
        itertools.chain.from_iterable(jailbreak_candidates)
    )
    target_outputs = list(
        itertools.chain.from_iterable(target_outputs)
    )
    
    # Update augmented_prompts and prompts_split
    for split in prompts_split:
        split_end = split["start"] + split["num_prompts"]

        split["start"] *= experiment_config.pair_breadth
        split["augs_per_prompt"] = experiment_config.pair_breadth * \
            experiment_config.pair_depth
    
    results = {
        "loaded": False,
        "experiment_name": experiment_name,
        "model_alias": model_alias,
        "prompts_split": prompts_split,
        "prompts": jailbreak_candidates,
        "outputs": target_outputs,
    }
    
    return results

def launch_generation_tasks(
    experiments: list,
    experiment_config_path: str,
    experiment_configs: dict,
    judge_config_path: str,
    judge_configs: dict,
    model_config_path: str,
    datasets: dict,
    comparisons: dict,
    augmentation_thresholds: dict,
    augmentation_names: dict,
    augmented_prompts: dict,
    outputs_ids: list,
    results_dir: str,
    results_data: dict,
    attack: str,
    logger: logging.Logger,
):
    """Starts generation tasks.

    Loads the configurations and datasets and stores them for later use. If 
    split_generation_tasks_by_augmentation is true in the experiment 
    configuration, starts a Ray task for each augmentation. Otherwise, starts a 
    single Ray task for the aggregation of all augmentations for each 
    (experiment, model) pair.

    Args:
        experiments: A list of experiment names to run.
        experiment_config_path: The path to the experiment configuration file.
        experiment_configs: A dictionary mapping experiment names to their 
            configurations.
        judge_config_path: The path to the judge configuration file.
        judge_configs: A dictionary mapping experiment names to a dictionary
            mapping judge aliases to their configurations.
        model_config_path: The path to the model configuration file.
        datasets: A dictionary mapping experiment names to their datasets.
        comparisons: A dictionary mapping experiment names to a list of their
            comparison configurations.
        augmentation_thresholds: A dictionary mapping experiment names to a
            dictionary mapping augmentation names to a list of thresholds.
        augmentation_names: A dictionary mapping experiment names to a 
            dictionary mapping augmentation indices to their names.
        augmented_prompts: A dictionary mapping experiment names to a dictionary
            mapping augmentation names to augmented prompts.
        outputs_ids: A list to add Ray task IDs to for generation.
        results_dir: The directory to look up existing evaluation results.
        results_data: A dictionary mapping experiment names to a 
            dictionary mapping model aliases to a dictionary mapping 
            augmentation names to a dictionary containing the results.
        attack: The type of attack to run.
        logger: The logger to use for logging.
    """
    added_experiments = []
    added_models = {}
    generation_task_queue = []

    cached_judge_configs = {}
    cached_model_configs = {}

    # Loop through each experiment in the experiments argument
    for experiment_name in experiments:
        # Skip experiment if already started    
        if experiment_name in added_experiments:
            logger.warning(
                DUPLICATE_EXPERIMENT_MESSAGE.format(
                    experiment_name=experiment_name
                ) 
            )

            continue

        # Load experiment configuration
        experiment_config = safe_load_experiment_config(
            experiment_name=experiment_name,
            config_path=experiment_config_path,
            logger=logger,
            log_config=True,
        )

        if experiment_config is None:
            continue
        else:
            # For PAIR attack, load attacker model and judge config
            attacker_model_config = None
            attacker_judge_config = None

            if attack == AttackType.PAIR.value:
                attacker_model_config = safe_load_model_config(
                    model_alias=experiment_config.pair_attack_model,
                    config_path=model_config_path,
                    cache=cached_model_configs,
                    experiment_name=experiment_name,
                    logger=logger,
                    log_config=True,
                )

                if attacker_model_config is None:
                    continue
                
                attacker_judge_config = safe_load_judge_config(
                    judge_alias=experiment_config.pair_attack_judge_model,
                    config_path=judge_config_path,
                    cache=cached_judge_configs,
                    experiment_name=experiment_name,
                    logger=logger,
                    log_config=True,
                )

            experiment_configs[experiment_name] = experiment_config

        # Load judge configurations
        for judge_alias in experiment_config.judges:
            judge_config = safe_load_judge_config(
                judge_alias=judge_alias,
                config_path=judge_config_path,
                cache=cached_judge_configs,
                experiment_name=experiment_name,
                logger=logger,
                log_config=True,
            )

            if judge_config is not None:
                judge_configs.setdefault(
                    experiment_name,
                    {},
                )[judge_alias] = judge_config
        
        # Skip experiment if no judge configurations are loaded
        if len(judge_configs.get(experiment_name, [])) == 0:
            continue

        # Load evaluation data
        dataset = safe_load_dataset(
            dataset_name=experiment_config.dataset,
            dataset_path=experiment_config.dataset_path,
            experiment_name=experiment_name,
            logger=logger,
        )
        attack_targets = {}

        if dataset is None:
            continue
        else:
            datasets[experiment_name] = dataset
        
        experiment_augmented_prompts = augmented_prompts.setdefault(
            experiment_name,
            {},
        )
        experiment_augmentation_names = augmentation_names.setdefault(
            experiment_name,
            {},
        )
        experiment_augmentation_thresholds = augmentation_thresholds.setdefault(
            experiment_name,
            {},
        )
        experiment_added_models = added_models.setdefault(
            experiment_name,
            [],
        )

        # Loop through each augmentation configuration
        for i, configuration in enumerate(experiment_config.augmentations):
            # Load augmentation configuration
            loaded_augmentation = safe_load_augmentation(
                config_dict=configuration,
                augmentation_index=i,
                experiment_name=experiment_name,
                logger=logger,
            )

            if loaded_augmentation is None:
                continue
            else:
                augmentation, augmentation_config = loaded_augmentation
            
            if not (augmentation_config.augmentation_samples == 1 or \
                experiment_config.output_samples == 1):
                logger.warning(
                    SAMPLES_ERROR_MESSAGE.format(
                        experiment_name=experiment_name,
                        augs_samples=augmentation_config.augmentation_samples,
                        output_samples=experiment_config.output_samples,
                    )
                )

                continue

            augmentation_name = get_augmentation_unique_name(
                augmentation_config=augmentation_config,
                output_samples=experiment_config.output_samples,
            )

            # Skip augmentation if already added
            if augmentation_name in experiment_augmentation_names.values():
                logger.warning(
                    DUPLICATE_AUGMENTATION_MESSAGE.format(
                        experiment_name=experiment_name,
                        augmentation_name=augmentation_name,
                    )
                )

                continue
            
            # Augment prompts
            prompts = safe_augment_prompts(
                dataset=dataset,
                augmentation=augmentation,
                augmentation_name=augmentation_name,
                augmentation_config=augmentation_config,
                random_seed=experiment_config.seed,
                experiment_name=experiment_name,
                logger=logger,
                log_augmentation=True,
            )

            if prompts is None:
                continue
            else:
                experiment_augmented_prompts[augmentation_name] = prompts
                experiment_augmentation_names[i] = augmentation_name
                experiment_augmentation_thresholds[augmentation_name] = \
                    augmentation_config.thresholds
                
                if attack == AttackType.PAIR.value:
                    attack_targets[augmentation_name] = [
                        [
                            dataset.get_target(i) for _ in range(
                                augmentation_config.augmentation_samples,
                            )
                        ] for i in range(len(dataset))
                    ]
            
        # Loop through each model in the experiment configuration
        for model_alias in experiment_config.models:
            # Skip model if already added
            if model_alias in experiment_added_models:
                logger.warning(
                    DUPLICATE_MODEL_MESSAGE.format(
                        experiment_name=experiment_name,
                        model_alias=model_alias,
                    ),
                )

                continue
            
            # Load model configuration
            model_config = safe_load_model_config(
                model_alias=model_alias,
                config_path=model_config_path,
                cache=cached_model_configs,
                experiment_name=experiment_name,
                logger=logger,
                log_config=True,
            )

            if model_config is None:
                continue
            
            # Queue up generation tasks for the model
            queue_model_generation_tasks(
                experiment_name=experiment_name,
                model_alias=model_alias,
                augmented_prompts=augmented_prompts[experiment_name],
                attack_targets=attack_targets,
                dataset=dataset,
                results_dir=results_dir,
                experiment_config=experiment_config,
                model_config=model_config,
                attacker_model_config=attacker_model_config,
                attacker_judge_config=attacker_judge_config,
                judge_configs=judge_configs,
                outputs_ids=outputs_ids,
                task_queue=generation_task_queue,
                added_experiments=added_experiments,
                experiment_added_models=experiment_added_models,
                results_data=results_data,
                attack=attack,
                logger=logger,
            )
    
    # Load comparisons configurations
    for experiment_name in added_experiments:
        comparisons_configs = \
            experiment_configs[experiment_name].comparisons

        for config_idx, configuration in enumerate(comparisons_configs):
            comparison_config = safe_load_comparisons_config(
                config_dict=configuration,
                config_idx=config_idx,
                experiment_name=experiment_name,
                logger=logger,
            )

            if comparison_config is not None:
                comparisons.setdefault(
                    experiment_name,
                    [],
                ).append(comparison_config)

    # Launch generation tasks
    for task in generation_task_queue:
        task_arguments = task["task_arguments"]
        task_launch_message = task["task_launch_message"]

        api = task_arguments["model_config"].api
        open_source = api == API.HUGGINGFACE.value or \
            api == API.VLLM.value

        if open_source:
            if attack == AttackType.STOCHASTIC_MONKEYS.value:
                outputs_ids.append(
                    generate_outputs.options(
                        num_gpus=task_arguments["model_config"].gpus,
                    ).remote(
                        **task_arguments,
                    )
                )
            elif attack == AttackType.PAIR.value:
                outputs_ids.append(
                    pair_attack.remote(
                        **task_arguments,
                    )
                )
        elif api == API.OPENAI.value:
            outputs_ids.append(
                generate_outputs_openai.remote(
                    prompts=task_arguments["prompts"],
                    prompts_split=task_arguments["prompts_split"],
                    model_alias=task_arguments["model_alias"],
                    model_config=task_arguments["model_config"],
                    experiment_name=task_arguments["experiment_name"],
                    experiment_config=task_arguments["experiment_config"],
                    results_dir=results_dir,
                )
            )
        else:
            logger.error(
                INVALID_API_ERROR.format(
                    api=api,
                )
            )

        logger.info(task_launch_message)
    
def launch_evaluation_tasks(
    outputs_ids: list,
    results_ids: list,
    judge_configs: dict,
    results_data: dict,
    experiment_configs: dict,
    datasets: dict,
    results_dir: str,
    logger: logging.Logger,
):
    """Starts evaluation tasks.
    
    Waits for generation to finish and then starts a Ray task to evaluate the
    generated outputs for each (experiment, model) pair.
    
    Args:
        outputs_ids: A list of Ray task IDs for generation tasks that are 
            currently running.
        results_ids: A list to add Ray task IDs to for evaluation.
        judge_configs: A dictionary mapping experiment names to a dictionary
            mapping judge aliases to their configurations.
        results_data: A dictionary mapping experiment names to a 
            dictionary mapping model aliases to a dictionary mapping 
            augmentation names to a dictionary containing the results. After 
            generation, this dictionary has the following keys:
                - summary: A dictionary of summary data for the results (empty 
                    until (k, gamma)-success scores are calculated)
                - results: A list of dictionaries containing the following keys:
                    - base_prompt: The base prompt.
                    - success_score: A dictionary of success scores for each 
                        threshold (empty until (k, gamma)-success scores are 
                        calculated).
                    - augmentations: A list of dictionaries containing the
                        following keys:
                        - prompt: The augmented prompt.
                        - output: The generated output.
        experiment_configs: A dictionary mapping experiment names to their
            configurations.
        datasets: A dictionary mapping experiment names to their datasets.
        augmented_prompts: A dictionary mapping experiment names to a dictionary
            mapping augmentation names to augmented prompts.
        results_dir: The directory to save results to.
        logger: The logger to use for logging.
    """

    while len(outputs_ids) > 0:
        outputs_done_id, outputs_ids = ray.wait(
            ray_waitables=outputs_ids,
            num_returns=1,
        )

        try:
            results = ray.get(outputs_done_id[0])

            loaded = results["loaded"]
            experiment_name = results["experiment_name"]
            model_alias = results["model_alias"]
            prompts_split = results["prompts_split"]
            augmented_prompts = results["prompts"]
            outputs = results["outputs"]
            
            if not loaded and "average_prompt_lengths" in results:
                average_prompt_lengths = results["average_prompt_lengths"]
            
            experiment_config = experiment_configs[experiment_name]

            current_dataset = datasets[experiment_name]

            judge_prompts = {}

            if not loaded and \
                not experiment_config.split_generation_tasks_by_augmentation:
                logger.info(
                    COMPLETED_GENERATION_TASK_MESSAGE.format(
                        experiment_name=experiment_name,
                        model_alias=model_alias,
                    )
                )
            
            for prompts_split_dict in prompts_split:
                augmentation_name = prompts_split_dict["augmentation_name"]
                start = prompts_split_dict["start"]
                num_prompts = prompts_split_dict["num_prompts"]
                augs_per_prompt = prompts_split_dict["augs_per_prompt"]

                results_dict = results_data.get(
                    experiment_name,
                    {},
                ).get(
                    model_alias,
                    {},
                ).get(
                    augmentation_name,
                    {},
                )

                curr_summary = results_dict.setdefault(
                    "summary",
                    {},
                )
                curr_results = results_dict.setdefault(
                    "results",
                    [],
                )

                if not loaded and "average_prompt_lengths" in results:
                    average_prompt_length = \
                        average_prompt_lengths[augmentation_name]
                    average_chars = average_prompt_length["chars"]
                    average_tokens = average_prompt_length["tokens"]

                    curr_summary["average_prompt_length"] = {
                        "characters": average_chars,
                        "tokens": average_tokens,
                    }

                    logger.info(
                        AVERAGE_INPUT_LENGTH.format(
                            experiment_name=experiment_name,
                            model_alias=model_alias,
                            augmentation_name=augmentation_name,
                            average_length_chars=average_chars,
                            average_length_tokens=average_tokens,
                        )
                    )

                if len(curr_results) > 0:
                    first_score = curr_results[0]["augmentations"][0].get(
                        "score",
                        {},
                    )
                else:
                    first_score = {}

                for i in range(num_prompts):
                    if not loaded:
                        prompt_results = {}

                        prompt_results["base_prompt"] = current_dataset[i]
                        prompt_results["success_score"] = {}
                        prompt_results["average_score"] = {}
                        prompt_results["augmentations"] = []

                    prompt_start = start + i * augs_per_prompt
                    prompt_end = prompt_start + augs_per_prompt
                    prompts = augmented_prompts[prompt_start:prompt_end]
                    prompt_outputs = outputs[prompt_start:prompt_end]

                    for prompt, output_list in zip(
                        prompts,
                        prompt_outputs,
                    ):
                        for output in output_list:
                            if not loaded:
                                prompt_results["augmentations"].append({
                                    "prompt": prompt,
                                    "output": output,
                                })

                            for judge_alias, judge_config in \
                                judge_configs[experiment_name].items():
                                if judge_alias not in first_score:
                                    judge_prompts.setdefault(
                                        judge_alias,
                                        [],
                                    ).append(
                                        judge_config.judge_prompt.format(
                                            user_prompt=current_dataset[i],
                                            model_output=output,
                                        )
                                    )

                    if not loaded:
                        results_dict["results"].append(prompt_results)
                    
                if not loaded:
                    if experiment_config.split_generation_tasks_by_augmentation:
                        logger.info(
                            COMPLETED_GENERATION_TASK_SPLIT_MESSAGE.format(
                                experiment_name=experiment_name,
                                model_alias=model_alias,
                                augmentation_name=augmentation_name,
                            )
                        )
            
                    results_data.setdefault(
                        experiment_name,
                        {},
                    ).setdefault(
                        model_alias,
                        {},
                    )[augmentation_name] = results_dict

                    save_results(
                        results_dict=results_dict,
                        experiment_name=experiment_name,
                        model_alias=model_alias,
                        augmentation_name=augmentation_name,
                        results_dir=results_dir,
                        logger=logger,
                    )
                
                if experiment_config.output_samples > 1:
                    prompts_split_dict["start"] *= \
                        experiment_config.output_samples
                    prompts_split_dict["augs_per_prompt"] = \
                        experiment_config.output_samples

            for judge_alias, prompts in judge_prompts.items():
                judge_config = judge_configs[experiment_name][judge_alias]

                results_ids.append(
                    generate_outputs.options(
                        num_gpus=judge_config.gpus,
                    ).remote(
                        system_prompts=[
                            "" for _ in prompts
                        ],
                        prompts=prompts,
                        prompts_split=prompts_split,
                        output_samples=1,
                        greedy=True,
                        temperature=0,
                        top_p=1,
                        model_alias=model_alias,
                        model_config=judge_config,
                        judge=True,
                        experiment_name=experiment_name,
                        experiment_config=experiment_config,
                        judge_alias=judge_alias,
                    )
            )
        except Exception as message:
            logger.error(
                ERROR_MESSAGE.format(
                    message=message,
                )
            )

def retrieve_evaluation_results_from_tasks(
    results_ids: list,
    judge_configs: dict,
    results_data: dict,
    results_dir: str,
    logger: logging.Logger,
):
    """Retrieves evaluation results from Ray tasks.

    Waits for an evaluation task to finish and then saves the results to a
    dictionary.

    Args:
        results_ids: A list of Ray task IDs for evaluation tasks that are
            currently running.
        judge_configs: A dictionary mapping experiment names to a dictionary
            mapping judge aliases to their configurations.
        results_data: A dictionary mapping experiment names to a 
            dictionary mapping model aliases to a dictionary mapping 
            augmentation names to a dictionary containing the results.
        results_dir: The directory to save results to.
        logger: The logger to use for logging.
    """

    while len(results_ids) > 0:
        results_done_id, results_ids = ray.wait(
            ray_waitables=results_ids,
            num_returns=1,
        )

        try:
            results = ray.get(results_done_id[0])

            experiment_name = results["experiment_name"]
            model_alias = results["model_alias"]
            prompts_split = results["prompts_split"]
            judge_outputs = results["outputs"]
            judge_alias = results["judge_alias"]

            judge_config = judge_configs[experiment_name][judge_alias]

            zero_label = judge_config.zero_label
            one_label = judge_config.one_label

            all_scores = [
                get_score(
                    output=output[0],
                    zero_label=zero_label,
                    one_label=one_label,
                ) for output in judge_outputs
            ]

            current_results = results_data[experiment_name][model_alias]

            for prompts_split_dict in prompts_split:
                augmentation_name = prompts_split_dict["augmentation_name"]
                start = prompts_split_dict["start"]
                num_prompts = prompts_split_dict["num_prompts"]
                augs_per_prompt = prompts_split_dict["augs_per_prompt"]
                end = start + num_prompts * augs_per_prompt
                
                split_scores = all_scores[start:end]

                results_dict = current_results[augmentation_name]

                for i in range(num_prompts):
                    prompt_results = results_dict["results"][i]

                    prompt_scores_start = i * augs_per_prompt
                    prompt_scores_end = prompt_scores_start + augs_per_prompt
                    prompt_scores = \
                        split_scores[prompt_scores_start:prompt_scores_end]

                    for augmentation, score in zip(
                        prompt_results["augmentations"],
                        prompt_scores,
                    ):
                        augmentation.setdefault(
                            "score",
                            {},
                        )[judge_alias] = score
                
                save_results(
                    results_dict=results_dict,
                    experiment_name=experiment_name,
                    model_alias=model_alias,
                    augmentation_name=augmentation_name,
                    results_dir=results_dir,
                    logger=logger,
                )
                
                logger.info(
                    COMPLETED_EVALUATION_MESSAGE.format(
                        experiment_name=experiment_name,
                        model_alias=model_alias,
                        augmentation_name=augmentation_name,
                    )
                )
        except Exception as message:
            logger.error(
                ERROR_MESSAGE.format(
                    message=message,
                )
            )

def add_k_gamma_success(
    results_data: dict,
    augmentation_thresholds: dict,
    results_dir: str,
    logger: logging.Logger,
):
    """Adds (k, gamma)-success scores and rates to evaluation results.

    Args:
        results_data: A dictionary mapping experiment names to a dictionary 
            mapping model aliases to a dictionary mapping augmentation names to 
            a dictionary containing the results.
        augmentation_thresholds: A dictionary mapping experiment names to a
            dictionary mapping augmentation names to a list of thresholds.
        results_dir: The directory to save results to.
        logger: The logger to use for logging.
    """

    results_iter = [
        (
            experiment_name,
            model_alias,
            augmentation_name,
            results_dict,
        )
        for experiment_name, experiment_results in results_data.items()
        for model_alias, model_results in experiment_results.items()
        for augmentation_name, results_dict in model_results.items()
    ]

    for current_results in results_iter:
        experiment_name, model_alias, augmentation_name, results_dict = \
            current_results
        
        summary = results_dict["summary"]
        results = results_dict["results"]

        thresholds = augmentation_thresholds[experiment_name][augmentation_name]

        total_success_score = {}
        overall_average_score = {}
        scored_prompts = {}
        unscored_prompts = {}

        # Calculate (k, gamma)-success scores
        for prompt_result_dict in results:
            augmentations_scores = [
                augmentation["score"] for augmentation in 
                    prompt_result_dict["augmentations"]
            ]

            # Filter out None values
            filtered_prompt_scores = {}

            for score_dict in augmentations_scores:
                for judge_alias, score in score_dict.items():
                    filtered_prompt_scores.setdefault(
                        judge_alias,
                        [],
                    )

                    if score is not None:
                        filtered_prompt_scores[judge_alias].append(score)

            for judge_alias, scores in filtered_prompt_scores.items():
                scored_prompts.setdefault(
                    judge_alias,
                    0,
                )
                unscored_prompts.setdefault(
                    judge_alias,
                    0,
                )

                if len(scores) > 0:
                    # Apply threshold to determine (k, gamma)-success
                    average_score = sum(scores) / len(scores)
                    threshold_success_score = [
                        int(average_score > threshold) 
                            for threshold in thresholds
                    ]

                    prompt_result_dict.setdefault(
                        "success_score",
                        {},
                    )[judge_alias] = {
                        str(threshold): threshold_success_score[i] 
                            for i, threshold in enumerate(thresholds)
                    }

                    prompt_result_dict.setdefault(
                        "average_score",
                        {},
                    )[judge_alias] = average_score

                    overall_average_score[judge_alias] = \
                        average_score + overall_average_score.get(
                            judge_alias,
                            0,
                        )

                    for i, score in enumerate(threshold_success_score):
                        total_success_score.setdefault(
                            judge_alias,
                            [0 for _ in thresholds],
                        )[i] += score
                    
                    scored_prompts[judge_alias] += 1
                else:
                    # If all scores are None, the prompt is unscored
                    prompt_result_dict.setdefault(
                        "success_score",
                        {},
                    )[judge_alias] = None

                    prompt_result_dict.setdefault(
                        "average_score",
                        {},
                    )[judge_alias] = None

                    unscored_prompts[judge_alias] += 1
            
        # Calculate (k, gamma)-success rates
        success_rate = {}

        for (judge_alias, judge_total_score), scored in zip(
            total_success_score.items(),
            scored_prompts.values(),
        ):
            success_rate[judge_alias] = [
                total / scored for total in judge_total_score
            ] if scored > 0 else None
        
        # Calculate overall averages
        for (judge_alias, average_score), scored in zip(
            overall_average_score.items(),
            scored_prompts.values(),
        ):
            overall_average_score[judge_alias] = \
                average_score / scored if scored > 0 else None

        summary["total_success_score"] = {
            judge_alias: {
                str(threshold): total 
                    for threshold, total in zip(thresholds, judge_total_score)
            } 
            for judge_alias, judge_total_score in total_success_score.items()
        }
        summary["scored_prompts"] = scored_prompts
        summary["unscored_prompts"] = unscored_prompts

        # Add (k, gamma)-success rates to summary
        for judge_alias, judge_rate in success_rate.items():
            summary.setdefault(
                "success_rate",
                {},
            )

            if judge_rate is not None:
                summary["success_rate"][judge_alias] = {
                    str(threshold): rate 
                        for threshold, rate in zip(thresholds, judge_rate)
                }

                for threshold, rate in zip(thresholds, judge_rate):
                    logger.info(
                        SUCCESS_RATE_MESSAGE.format(
                            experiment_name=experiment_name,
                            model_alias=model_alias,
                            augmentation_name=augmentation_name,
                            judge_alias=judge_alias,
                            threshold=float(threshold),
                            success_rate=rate,
                        )
                    )
            else:
                summary["success_rate"][judge_alias] = None
            
            summary["average_score"] = overall_average_score

        save_results(
            results_dict=results_dict,
            experiment_name=experiment_name,
            model_alias=model_alias,
            augmentation_name=augmentation_name,
            results_dir=results_dir,
            logger=logger,
        )

def compare_results(
    comparisons: dict,
    results_data: dict,
    augmentation_names: dict,
    experiment_config_path: str,
    results_dir: str,
    logger: logging.Logger,
):
    """Compares results.

    Args:
        comparisons: A dictionary mapping experiment names to a list of their
            comparison configurations.
        results_data: A dictionary mapping experiment names to a 
            dictionary mapping model aliases to a dictionary mapping 
            augmentation names to a dictionary containing the results.
        augmentation_names: A dictionary mapping augmentation indices to their 
            names for the current experiment.
        experiment_config_path: The path to the experiment configuration file.
        results_dir: The directory to save results to.
        logger: The logger to use for logging.
    """

    def compare(
        current_results: dict,
        other_results: dict,
        other_experiment_name: str,
        other_model_alias: str,
        other_augmentation_name: str,
        threshold: float,
        other_threshold: float,
    ):
        """Compares current results to other results.

        Adds comparison data to the current results.

        Args:
            current_results: The current results to compare.
            other_results: The other results to compare.
            other_experiment_name: The name of the other experiment.
            other_model_alias: The alias of the other model.
            other_augmentation_name: The name of the other augmentation.
            threshold: The threshold for the current results.
            other_threshold: The threshold for the other results.
        """

        current_summary = current_results["summary"]
        other_summary = other_results["summary"]

        current_success_rate = {
            judge_alias: rate.get(str(threshold), None)
                for judge_alias, rate in current_summary["success_rate"].items()
        }
        other_success_rate = {
            judge_alias: rate.get(str(other_threshold), None)
                for judge_alias, rate in other_summary["success_rate"].items()
        }

        # Calculate success rate gain
        for judge_alias, current_rate in current_success_rate.items():
            other_rate = other_success_rate.get(
                judge_alias,
                None,
            )

            if current_rate is not None and other_rate is not None:
                current_results["summary"].setdefault(
                    "success_rate_gain",
                    {},
                ).setdefault(
                    str(threshold),
                    {},
                ).setdefault(
                    other_experiment_name,
                    {},
                ).setdefault(
                    other_model_alias,
                    {},
                ).setdefault(
                    other_augmentation_name,
                    {},
                ).setdefault(
                    str(other_threshold),
                    {},
                )[judge_alias] = current_rate - other_rate
        
        overall_average_score_gain_computed = judge_alias in \
            current_results["summary"].get(
                "average_score_gain",
                {},
            ).get(
                other_experiment_name,
                {},
            ).get(
                other_model_alias,
                {},
            ).get(
                other_augmentation_name,
                {},
            )

        # Calculate overall average score gain
        if not overall_average_score_gain_computed:
            current_average_score = current_summary["average_score"]
            other_average_score = other_summary["average_score"]

            # Calculate overall average score gains
            for judge_alias, current_average in current_average_score.items():
                other_average = other_average_score.get(
                    judge_alias,
                    None,
                )

                if current_average is not None and other_average is not None:
                    current_results["summary"].setdefault(
                        "average_score_gain",
                        {},
                    ).setdefault(
                        other_experiment_name,
                        {},
                    ).setdefault(
                        other_model_alias,
                        {},
                    ).setdefault(
                        other_augmentation_name,
                        {},
                    )[judge_alias] = current_average - other_average
    
    def get_other_augmentation_name(
        other_experiment_name: str,
        experiment_config_path: str,
        other_augmentation_idx: int,
        logger: logging.Logger,
    ) -> str:
        """Returns the name of the other augmentation.

        Args:
            other_experiment_name: The name of the other experiment.
            experiment_config_path: The path to the experiment configuration
                file.
            other_augmentation_idx: The index of the other augmentation.
            logger: The logger to use for logging.
        """

        # Load other experiment configuration
        other_experiment_config = safe_load_experiment_config(
            experiment_name=other_experiment_name,
            config_path=experiment_config_path,
            logger=logger,
            log_config=False,
        )

        if other_experiment_config is None:
            return None

        # Check if index is valid
        valid_idx = other_augmentation_idx >= 0 and \
            other_augmentation_idx < len(other_experiment_config.augmentations)
        
        if valid_idx:
            other_augmentation_config_dict = \
                other_experiment_config.augmentations[other_augmentation_idx]
        else:
            return None

        # Load other augmentation configuration
        _, other_augmentation_config = safe_load_augmentation(
            config_dict=other_augmentation_config_dict,
            augmentation_index=other_augmentation_idx,
            experiment_name=other_experiment_name,
            logger=logger,
        )
        
        other_augmentation_name = get_augmentation_unique_name(
            augmentation_config=other_augmentation_config,
            output_samples=other_experiment_config.output_samples,
        )

        return other_augmentation_name
    
    def retrieve_comparison_iter(
        current_config: Config,
        other_config: Config,
    ) -> list:
        """Returns an iterator for comparing results.

        Args:
            current_config: The current comparison configuration.
            other_config: The other comparison configuration.
        """

        def retrieve_comparisons(
            current: list,
            other: list,
            comparison_type: ComparisonType,
        ) -> zip | itertools.product:
            """Returns a comparison iterator between two lists.

            Args:
                current: The current list of configurations.
                other: The other list of configurations.
                comparison_type: The type of comparison to perform.
            """

            comparison_iter = []
            same_length = len(current) == len(other)

            if comparison_type == ComparisonType.ELEMENTWISE and same_length:
                comparison_iter = zip(current, other)
            elif comparison_type == ComparisonType.CARTESIAN:
                comparison_iter = itertools.product(current, other)
            
            return comparison_iter

        models_comparison = retrieve_comparisons(
            current_config.models,
            other_config.models,
            other_config.models_comparison,
        )
        augmentations_comparison = retrieve_comparisons(
            current_config.augmentations,
            other_config.augmentations,
            other_config.augmentations_comparison,
        )
        thresholds_comparison = retrieve_comparisons(
            current_config.thresholds,
            other_config.thresholds,
            other_config.thresholds_comparison,
        )

        comparisons_iter = itertools.product(
            models_comparison,
            augmentations_comparison,
            thresholds_comparison,
        )

        return comparisons_iter

    # Loop through all experiments
    for experiment_name, experiment_comparisons in comparisons.items():
        experiment_results = results_data.get(experiment_name, {})
        current_augmentations_names = augmentation_names.get(
            experiment_name,
            {},
        )

        # Loop through all comparison configurations in experiment
        for current_config in experiment_comparisons:
            comparisons = []

            # Create comparison list
            for other_config in current_config.other_results:
                other_comparisons = retrieve_comparison_iter(
                    current_config=current_config,
                    other_config=other_config,
                )

                for comparison in other_comparisons: 
                    comparisons.append(
                        (
                            other_config.experiment_name,
                            comparison,
                        )
                    )

            # Loop through all comparisons
            for other_experiment_name, comparison in comparisons:
                (
                    (model_alias, other_model_alias),
                    (augmentation_idx, other_augmentation_idx),
                    (threshold, other_threshold),
                ) = comparison
                
                augmentation_name = current_augmentations_names.get(
                    augmentation_idx,
                    None,
                )
                other_augmentation_name = get_other_augmentation_name(
                    other_experiment_name=other_experiment_name,
                    experiment_config_path=experiment_config_path,
                    other_augmentation_idx=other_augmentation_idx,
                    logger=logger,
                )

                if augmentation_name is None or other_augmentation_name is None:
                    continue

                current_results = experiment_results.get(
                    model_alias,
                    {},
                ).get(
                    augmentation_name,
                    None,
                )

                # Check if other results is in the same experiment
                other_results = results_data.get(
                    other_experiment_name,
                    {},
                ).get(
                    other_model_alias,
                    {},
                ).get(
                    other_augmentation_name,
                    None,
                )

                # Check if other results is in the results directory
                if other_results is None:
                    other_results_path = Path(
                        results_dir,
                        other_experiment_name,
                        other_model_alias,
                        RESULTS_FILENAME.format(
                            augmentation_name=other_augmentation_name,
                        ),
                    )

                    exists = other_results_path.exists() and \
                        other_results_path.is_file()

                    if exists:
                        with other_results_path.open(
                            LOOKUP_RESULTS_FILE_OPEN_MODE
                        ) as f:
                            other_results = json.load(f)

                # Perform comparison
                if current_results is not None and other_results is not None:
                    compare(
                        current_results=current_results,
                        other_results=other_results,
                        other_experiment_name=other_experiment_name,
                        other_model_alias=other_model_alias,
                        other_augmentation_name=other_augmentation_name,
                        threshold=threshold,
                        other_threshold=other_threshold,
                    )

                    save_results(
                        results_dict=current_results,
                        experiment_name=experiment_name,
                        model_alias=model_alias,
                        augmentation_name=augmentation_name,
                        results_dir=results_dir,
                        logger=logger,
                        log_save=False,
                    )
    
def save_results(
    results_dict: dict,
    experiment_name: str,
    model_alias: str,
    augmentation_name: str,
    results_dir: str,
    logger: logging.Logger,
    log_save: bool = True,
):
    """Saves evaluation results to files.

    Args:
        results_dict: A dictionary of results to save.
        experiment_name: The name of the current experiment.
        model_alias: The alias of the current model.
        augmentation_name: The name of the current augmentation.
        results_dir: The directory to save the results to.
        logger: The logger to use for logging.
        log_save: Whether to log the save operation.
    """

    current_results_dir = Path(
        results_dir,
        experiment_name,
        model_alias,
    )
    current_results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_filename = RESULTS_FILENAME.format(
        augmentation_name=augmentation_name,
    )
    results_path = current_results_dir / results_filename
    results_path.touch(exist_ok=True)

    with results_path.open(SAVE_RESULTS_FILE_OPEN_MODE) as f:
        json.dump(
            results_dict,
            f,
            indent=RESULTS_FILE_INDENT,
        )
    
    if log_save:
        logger.info(
            SAVED_RESULTS_MESSAGE.format(
                experiment_name=experiment_name,
                model_alias=model_alias,
                augmentation_name=augmentation_name,
                save_path=results_path,
            )
        )

def main():
    """Main function for the script."""

    torch.set_default_device(DEFAULT_DEVICE)

    args = get_args()
    logger = get_logger(log_path=args.log_path)
    
    experiment_configs = {}
    judge_configs = {}
    datasets = {}
    comparisons = {}
    augmentation_thresholds = {}
    augmentation_names = {}
    augmented_prompts = {}
    results_data = {}
    outputs_ids = []
    results_ids = []

    if args.attack == AttackType.PAIR.value:  
        if len(args.experiments) > 1:
            raise ValueError(
                PAIR_ATTACK_EXPERIMENTS_ERROR.format(
                    num_experiments=len(args.experiments),
                )
            )
    
    launch_generation_tasks(
        experiments=args.experiments,
        experiment_config_path=args.experiment_config_path,
        experiment_configs=experiment_configs,
        judge_config_path=args.judge_config_path,
        judge_configs=judge_configs,
        model_config_path=args.model_config_path,
        datasets=datasets,
        comparisons=comparisons,
        augmentation_thresholds=augmentation_thresholds,
        augmentation_names=augmentation_names,
        augmented_prompts=augmented_prompts,
        outputs_ids=outputs_ids,
        results_dir=args.results_dir,
        results_data=results_data,
        attack=args.attack,
        logger=logger,
    )

    launch_evaluation_tasks(
        outputs_ids=outputs_ids,
        results_ids=results_ids,
        datasets=datasets,
        judge_configs=judge_configs,
        results_data=results_data,
        experiment_configs=experiment_configs,
        results_dir=args.results_dir,
        logger=logger,
    )

    retrieve_evaluation_results_from_tasks(
        results_ids=results_ids,
        judge_configs=judge_configs,
        results_data=results_data,
        results_dir=args.results_dir,
        logger=logger,
    )

    add_k_gamma_success(
        results_data=results_data,
        augmentation_thresholds=augmentation_thresholds,
        results_dir=args.results_dir,
        logger=logger,
    )

    compare_results(
        comparisons=comparisons,
        results_data=results_data,
        augmentation_names=augmentation_names,
        experiment_config_path=args.experiment_config_path,
        results_dir=args.results_dir,
        logger=logger
    )

    logger.info(COMPLETED_EVALUATIONS_MESSAGE)
    
if __name__ == "__main__":
    main()