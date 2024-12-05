import json
import os
import random
from tqdm import tqdm

random.seed(0)

standard_augmentations = [
    "none_samples_1.json",
    "string_suffix_percentage_0_05_samples_25.json",
    "string_prefix_percentage_0_05_samples_25.json",
    "string_anywhere_percentage_0_05_samples_25.json",
    "character_edit_percentage_0_05_samples_25.json",
    "character_insertion_percentage_0_05_samples_25.json",
    "character_deletion_percentage_0_05_samples_25.json",
]
temperature_augmentations = [
    "none_samples_25.json"
] + standard_augmentations[1:]
length_generalization_augmentations = [
    "string_suffix_length_5_samples_25.json",
    "string_suffix_length_10_samples_25.json",
    "string_suffix_length_15_samples_25.json",
    "string_suffix_length_20_samples_25.json",
    "string_suffix_length_25_samples_25.json",
    "string_suffix_length_30_samples_25.json",
    "string_suffix_length_35_samples_25.json",
    "string_suffix_length_40_samples_25.json",
    "string_suffix_length_45_samples_25.json",
    "string_suffix_length_50_samples_25.json",
]

experiments = {
    "all_llama_samples_25": standard_augmentations,
    "all_mistral_samples_25": standard_augmentations,
    "all_phi_samples_25": standard_augmentations,
    "all_qwen_samples_25": standard_augmentations,
    "defense_string_fixed_length": length_generalization_augmentations,
    "generation_0_7": temperature_augmentations,
    "generation_1": temperature_augmentations,
    "quantization_w4a16": standard_augmentations,
    "quantization_w8a8": standard_augmentations,
}

save_dir = "manual_label_splits"

os.makedirs(save_dir, exist_ok=True)

pos_pred_data = []
neg_pred_data = []
humans = 4
num_subsplits = 80
splits = [[] for _ in range(humans)]

print("Loading data...")

for experiment, augmentations in tqdm(experiments.items(), desc="Experiment"):
    file_dir = os.path.join("results", experiment)

    for model in tqdm(os.listdir(file_dir), desc="Model", leave=False):
        for augmentation in tqdm(
            augmentations, desc="Augmentation", leave=False
        ):
            file_path = os.path.join(file_dir, model, augmentation)

            augmentation_name = augmentation.split(".")[0]

            with open(file_path, "r") as f:
                data = json.load(f)

                for sample in data["results"]:
                    sample["success_score"]["human"] = {}

                    sample_data = {
                        "experiment": experiment,
                        "model": model,
                        "augmentation": augmentation_name,
                        "data": sample
                    }

                    score_dict = sample["success_score"]["sorry_bench_ft_mistral_7b_instruct"]

                    if score_dict is not None:
                        score = score_dict["0"]

                        if score == 1:
                            pos_pred_data.append(sample_data)
                        elif score == 0:
                            neg_pred_data.append(sample_data)

print("Positive predicted data:", len(pos_pred_data))
print("Negative predicted data:", len(neg_pred_data))
print("Total data:", len(pos_pred_data) + len(neg_pred_data))

random.shuffle(pos_pred_data)
random.shuffle(neg_pred_data)

print("Creating splits...")

while len(pos_pred_data) > 0 or len(neg_pred_data) > 0:
    for i in range(humans):
        if len(pos_pred_data) > 0:
            splits[i].append(pos_pred_data.pop())
        
        if len(neg_pred_data) > 0:
            splits[i].append(neg_pred_data.pop())

print("Saving splits...")

for i, split in enumerate(splits):
    subsplits = [
        split[j:j + (len(split) // num_subsplits)]
            for j in range(0, len(split), len(split) // num_subsplits)
    ]
    subsplit_sizes = [len(subsplit) for subsplit in subsplits]

    print(f"Split {i} size:", len(split))
    print(
        f"Split {i} subsplit sizes:",
        subsplit_sizes,
        f"(sum: {sum(subsplit_sizes)})"
    )

    for j, subsplit in enumerate(subsplits):
        save_path = f"{save_dir}/manual_label_split_{i}_subsplit_{j}.json"

        with open(save_path, "w") as f:
            save_dict = {
                "human": i,
                "sample": subsplit,
            }

            json.dump(save_dict, f, indent=4)