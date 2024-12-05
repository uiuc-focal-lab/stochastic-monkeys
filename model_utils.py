from enum import StrEnum, auto
import time
import json
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    AutoTokenizer,
    BatchEncoding,
)
from vllm import LLM, SamplingParams

class API(StrEnum):
    HUGGINGFACE = auto()
    OPENAI = auto()
    VLLM = auto()

MODEL_TYPE_ERROR = (
    "Model is of type {model_type}, but expected type LLM or PreTrainedModel."
)
INVALID_API_ERROR = "Invalid API: {api}. Must be 'vllm' or 'huggingface'."
OPENAI_INPUT_FILE_DIR = "input_files"
OPENAI_INPUT_FILE_NAME = "openai_input_{timestamp}.jsonl"

def load_model(
    name_or_path: str,
    api: str,
    **model_kwargs: dict,
) -> LLM | PreTrainedModel:
    """Returns an LLM.

    The model may either be loaded using the VLLM API or the HuggingFace API, 
    depending on the value of the api parameter.
    
    Args:
        name_or_path: The HuggingFace Hub model ID or path to 
            the model.
        api: The API to use ("vllm" or "huggingface").
        model_kwargs: Keyword arguments for loading the model. The provided
            keywords should be appropriate for the API being used to load the 
            model.
    
    Raises:
        ValueError: If the value of api is invalid.
    """

    if api == API.VLLM.value:
        model = LLM(
            model=name_or_path,
            **model_kwargs,
        )
    elif api == API.HUGGINGFACE.value:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=name_or_path,
            **model_kwargs,
        )

        model = model.cuda()
    else:
        raise ValueError(
            INVALID_API_ERROR.format(
                api=api
            )
        )
    
    return model

def load_tokenizer(
    model_name_or_path: str,
    **tokenizer_kwargs: dict,
) -> AutoTokenizer:
    """Returns a tokenizer.

    Args:
        model_name_or_path: The HuggingFace Hub model ID or path to 
            the model whose tokenizer to load.
        tokenizer_kwargs: Keyword arguments for loading the tokenizer.
    """

    if "padding_side" not in tokenizer_kwargs:
        tokenizer_kwargs["padding_side"] = "left"

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        **tokenizer_kwargs,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def get_generation_inputs(
    system_prompts: list[str],
    chat_history: list,
    prompts: list[str],
    prefill: str,
    tokenizer: AutoTokenizer,
    api: str,
    apply_chat_template: bool,
) -> list[list[int]] | BatchEncoding:
    """Returns the tokenized inputs to be used for generation.

    The list of prompts should contain untemplated user prompts, as this
    function will apply the tokenizer's chat template to the prompts.

    Args:
        system_prompts: A list of system prompts to use for each prompt.
        chat_history: The chat history so far preceding the user prompts 
            specified in the prompts argument. Should end in an assistant
            response.
        prompts: A list of the latest user prompts for the model.
        prefill: A prefill string for the assistant response.
        tokenizer: The tokenizer to use.
        api: The API used to load the model ("vllm" or "huggingface").
        apply_chat_template: Whether to apply the default chat template to the 
            prompts.
    """

    if apply_chat_template:
        conversations = [
            [] for prompt in prompts
        ]

        for i, (conversation, system_prompt, history, prompt) in enumerate(
            zip(
                conversations,
                system_prompts,
                chat_history,
                prompts,
            )
        ):
            if system_prompt != "":
                conversation.append(
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                )
            
            for j, content in enumerate(history):
                conversation.append(
                    {
                        "role": "user" if j % 2 == 0 else "assistant",
                        "content": content,
                    }
                )
            
            conversation.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

        inputs = tokenizer.apply_chat_template(
            conversation=conversations,
            padding=(api == API.HUGGINGFACE.value),
            add_generation_prompt=True,
            return_dict=(api == API.HUGGINGFACE.value),
            return_tensors=(None if api == API.VLLM.value else "pt"),
        )

        if prefill != "":
            prefill = tokenizer(
                text=prefill,
                add_special_tokens=False,
                return_attention_mask=(api == API.HUGGINGFACE.value),
                return_tensors=(None if api == API.VLLM.value else "pt"),
            )

            if api == API.HUGGINGFACE.value:
                prefill_input_ids = prefill["input_ids"].repeat(
                    inputs["input_ids"].size(0),
                    1,
                )
                prefill_attention_mask = prefill["attention_mask"].repeat(
                    inputs["attention_mask"].size(0),
                    1,
                )

                inputs["input_ids"] = torch.cat(
                    (inputs["input_ids"], prefill_input_ids),
                    dim=1,
                )
                inputs["attention_mask"] = torch.cat(
                    (inputs["attention_mask"], prefill_attention_mask),
                    dim=1,
                )
            else:
                for sequence in inputs:
                    sequence.extend(prefill["input_ids"])
        
        if api == API.HUGGINGFACE.value:
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].cuda()
    else:
        inputs = tokenizer(
            text=prompts,
            padding=(api == API.HUGGINGFACE.value),
            add_special_tokens=False,
            return_attention_mask=(api == API.HUGGINGFACE.value),
            return_tensors=(None if api == API.VLLM.value else "pt"),
        )

        if api == API.VLLM.value:
            inputs = inputs["input_ids"]
    
    return inputs

def create_openai_input_file(
    model_name: str,
    prompts: list[str],
    results_dir: str,
    experiment_name: str,
    experiment_config: dict,
    model_alias: str,
) -> str:
    """Creates an input file for OpenAI.

    Args:
        model_name: The name of the model being evaluated.
        prompts: A list of prompts to generate outputs for.
        results_dir: The directory to save results to. Used to save the input 
            file.
        experiment_name: The name of the experiment.
        experiment_config: The configuration for the experiment.
        model_alias: The alias for the model being evaluated.
    
    Returns:
        The path to the input file.
    """

    current_results_dir = Path(
        results_dir,
        experiment_name,
        model_alias,
        OPENAI_INPUT_FILE_DIR,
    )
    current_results_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    input_file_path = current_results_dir / OPENAI_INPUT_FILE_NAME.format(
        timestamp=str(time.time()).replace(".", "_"),
    )

    greedy = experiment_config.greedy

    with open(input_file_path, "w") as input_file:
        for i, prompt in enumerate(prompts):
            request = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "max_completion_tokens": \
                        experiment_config.max_new_tokens,
                    "seed": experiment_config.seed,
                    "temperature": 0 if greedy else \
                        experiment_config.temperature,
                    "top_p": 1 if greedy else \
                        experiment_config.top_p,
                },
            }

            input_file.write(json.dumps(request) + "\n")
    
    return input_file_path

def get_generation_outputs(
    model: LLM | PreTrainedModel,
    tokenizer: AutoTokenizer,
    inputs: list[list[int]] | BatchEncoding,
    output_samples: int,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_p: float,
    random_seed: int,
    vllm_use_tqdm: bool = False,
) -> list[str]:
    """Returns the generated outputs for the given inputs.

    If the model is of type LLM (i.e. VLLM is used), the inputs should be of 
    type list[list[int]]. Otherwise, the inputs should be of type BatchEncoding.

    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer to use.
        inputs: The tokenized inputs to the model.
        output_samples: The number of output samples to generate.
        max_new_tokens: The maximum number of tokens to generate.
        greedy: Whether to use greedy decoding.
        temperature: Temperature scaling factor for sampling. Ignored if greedy
            is True.
        top_p: The cumulative probability of top tokens to sample from. Ignored
            if greedy is True.
        random_seed: The random seed to use for generation.
        vllm_use_tqdm: Whether to use tqdm for VLLM generation.
    """

    torch.manual_seed(random_seed)

    if isinstance(model, LLM):
        sampling_params = SamplingParams(
            n=output_samples if not greedy else 1,
            temperature=(0 if greedy else temperature),
            top_p=(1 if greedy else top_p), # Use default if greedy
            max_tokens=max_new_tokens,
            seed=random_seed,
        )

        outputs = model.generate(
            prompt_token_ids=inputs,
            sampling_params=sampling_params,
            use_tqdm=vllm_use_tqdm,
        )

        outputs = [
            [
                o.text.strip() for o in output.outputs
            ] for output in outputs
        ]
        
    elif isinstance(model, PreTrainedModel):
        tokenized_output = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=(not greedy),
            temperature=(1 if greedy else temperature), # Use default if greedy
            top_p=(1 if greedy else top_p), # Use default if greedy
            num_return_sequences=output_samples if not greedy else 1,
        )[:, inputs["input_ids"].size(1):]

        outputs = tokenizer.batch_decode(
            sequences=tokenized_output,
            skip_special_tokens=True,
        )

        outputs = [output.strip() for output in outputs]

        outputs = [
            outputs[i:i + output_samples] 
                for i in range(0, len(outputs), output_samples)
        ]
    else:
        raise ValueError(
            MODEL_TYPE_ERROR.format(
                model_type=type(model).__name__
            )
        )

    return outputs