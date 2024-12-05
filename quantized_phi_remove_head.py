from safetensors import safe_open
from safetensors.torch import save_file

file_name = "quantized_models/Phi-3-small-8k-instruct-W4A16/model-00002-of-00002.safetensors"

tensors = {}

with safe_open(file_name, framework="pt", device=0) as f:
    for k in f.keys():
        if k == "lm_head.weight":
            continue

        tensors[k] = f.get_tensor(k)

save_file(tensors, file_name)