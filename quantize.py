from pathlib import Path
from config.config_utils import load_model_config
import argparse
import torch
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import (
    SparseAutoModelForCausalLM,
    oneshot,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationType,
    QuantizationStrategy,
    QuantizationScheme,
)
from transformers import AutoTokenizer

torch.set_default_device("cuda")

parser = argparse.ArgumentParser()

parser.add_argument(
    "model_alias",
    type=str,
    help="The alias of the model to quantize.",
)
parser.add_argument(
    "scheme",
    type=str,
    help="The quantization scheme to use.",
    choices=["W8A8", "W4A16"],
)
parser.add_argument(
    "--models_dir",
    type=str,
    default="./quantized_models/",
    help="The directory to save the quantized model and tokenizer.",
)
parser.add_argument(
    "--config_path",
    type=str,
    default="./config/models.toml",
    help="The path to the model configuration file.",
)

args = parser.parse_args()

model_config = load_model_config(
    name=args.model_alias,
    config_path=args.config_path,
)
model_name_or_path = model_config.name_or_path
trust_remote_code = model_config.model_kwargs.get("trust_remote_code", False)

model = SparseAutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=trust_remote_code,
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=trust_remote_code,
)

W8A8 = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    ),
)

W4A16 = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    ),
)

schemes = {
    "W8A8": W8A8,
    "W4A16": W4A16,
}

recipe = QuantizationModifier(
    config_groups={
        "group_0": schemes[args.scheme],
    },
    ignore=["lm_head"],
)

save_path = Path(
    args.models_dir,
    model_name_or_path.split("/")[-1].replace(".", "_") + f"-{args.scheme}",
)

oneshot(
    model=model,
    recipe=recipe,
    output_dir=save_path,
)

tokenizer.save_pretrained(save_path)