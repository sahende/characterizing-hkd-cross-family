import os
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from torch.utils.data import DataLoader

from config import Config


# Linear CKA following Kornblith et al. (2019).
def cka(X, Y):
    """
    Linear Centered Kernel Alignment (Kornblith et al., 2019).

    Args:
        X: (n_samples, d1) tensor
        Y: (n_samples, d2) tensor
    Returns:
        CKA similarity in [0, 1]
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    X_gram = X @ X.T
    Y_gram = Y @ Y.T

    n = X_gram.shape[0]
    H = torch.eye(n, device=X.device) - torch.ones(n, n, device=X.device) / n
    X_gram_centered = H @ X_gram @ H
    Y_gram_centered = H @ Y_gram @ H

    hsic_xy = (X_gram_centered * Y_gram_centered).sum()
    hsic_xx = (X_gram_centered * X_gram_centered).sum()
    hsic_yy = (Y_gram_centered * Y_gram_centered).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy + 1e-12)
    if denom == 0:
        return 0.0
    return float(hsic_xy / denom)


# Jensen--Shannon divergence.
def jensen_shannon_divergence(P, Q):
    """
    Jensen-Shannon Divergence between two soft-label distributions.

    JSD is the symmetric, bounded version of KL divergence. With the
    natural logarithm it is bounded in [0, log 2].

    Args:
        P: (n, C) tensor of probabilities
        Q: (n, C) tensor of probabilities
    Returns:
        float, mean JSD over the batch (in nats)
    """
    eps = 1e-12
    P = P + eps
    Q = Q + eps
    M = 0.5 * (P + Q)
    kl_pm = (P * (P.log() - M.log())).sum(-1).mean()
    kl_qm = (Q * (Q.log() - M.log())).sum(-1).mean()
    jsd = 0.5 * (kl_pm + kl_qm)
    return float(jsd)


# Dendrogram construction.
def build_dendrogram(cka_matrix, labels, title, save_path):
    """
    Build and save a dendrogram from a CKA similarity matrix.

    CKA is a similarity in [0, 1]; we convert it to a distance as 1 - CKA,
    zero out the diagonal, symmetrize, and pass the condensed upper
    triangle to linkage. This avoids the ClusterWarning and ensures the
    dendrogram reflects the intended distance geometry.
    """
    distance_matrix = 1.0 - cka_matrix
    distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0

    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="average")

    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=labels, orientation="left")
    plt.title(title)
    plt.xlabel("Distance (1 - CKA)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# Model and tokenizer loading.
def get_model_and_tokenizer(base_model, task_name, path, device):
    """Load tokenizer and model directly bypassing Config constants."""
    model_dir_mapping = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "electra-base": "electra-base",
        "gpt2": "gpt2",
        "gpt-neo-125M": "gpt-neo-125M",
        "opt-125m": "opt-125m",
    }
    registry_key = model_dir_mapping.get(base_model, base_model)
    source_dir = (
        Path(Config.PROJECT_ROOT)
        / "data"
        / "models"
        / Config.MODEL_REGISTRY[registry_key]["directory"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        source_dir, local_files_only=Config.local_files_only()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForSequenceClassification.from_pretrained(
        source_dir, num_labels=2, local_files_only=Config.local_files_only()
    )
    if model.config.pad_token_id is None and getattr(
        model.config, "eos_token_id", None
    ) is not None:
        model.config.pad_token_id = model.config.eos_token_id

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer


# Representation extraction from the final layer and final four layers.
def get_representations_and_logits(
    model, tokenizer, dataset, device, batch_size=16, is_single_sentence=True
):
    """
    Extract mean-pooled hidden representations and logits.

    Returns:
        reps_last: (n, d) last-layer mean-pooled representations
        reps_avg4: (n, d) average of last 4 layers (mean-pooled)
        logits:    (n, num_labels)
    """

    def preprocess(examples):
        if is_single_sentence:
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=Config.MAX_SEQ_LENGTH,
            )
        else:
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=Config.MAX_SEQ_LENGTH,
            )

    tokenized = dataset.map(
        preprocess, batched=True, remove_columns=dataset.column_names
    )
    format_columns = ["input_ids", "attention_mask"]
    if "token_type_ids" in tokenized.features:
        format_columns.append("token_type_ids")

    tokenized.set_format(type="torch", columns=format_columns)
    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=False)

    all_reps_last = []
    all_reps_avg4 = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting", leave=False):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs, output_hidden_states=True)

            logits = outputs.logits
            hidden_states = outputs.hidden_states

            mask = inputs["attention_mask"].unsqueeze(-1).float()
            mask_sum = torch.clamp(mask.sum(1), min=1e-9)

            # Mean-pool the final hidden layer.
            last_hidden = hidden_states[-1]
            pooled_last = (last_hidden * mask).sum(1) / mask_sum

            # Mean-pool the average of the final four hidden layers.
            last4 = torch.stack(hidden_states[-4:], dim=0)  # (4, B, T, D)
            last4_mean = last4.mean(0)  # (B, T, D)
            pooled_avg4 = (last4_mean * mask).sum(1) / mask_sum

            all_reps_last.append(pooled_last.cpu())
            all_reps_avg4.append(pooled_avg4.cpu())
            all_logits.append(logits.cpu())

    return (
        torch.cat(all_reps_last, dim=0),
        torch.cat(all_reps_avg4, dim=0),
        torch.cat(all_logits, dim=0),
    )


# Analysis for one task, dataset-size, and seed group.
def run_analysis_for_group(
    task_name, dataset_size_str, seed_str, teacher_paths, device
):
    """Run proximity analysis for a specific (task, size, seed) group."""
    print(f"\n{'='*70}")
    print(
        f"  ANALYSIS FOR: {task_name.upper()} | Size: {dataset_size_str} | Seed: {seed_str}"
    )
    print(f"{'='*70}")

    is_single_sentence = task_name in ["sst2", "cola"]

    import socket

    def has_internet(host="8.8.8.8", port=53, timeout=3):
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except OSError:
            return False

    if Config.DATA_OFFLINE or not has_internet():
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    dataset_path = os.path.join(Config.DATA_DIR, task_name)
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}. Skipping.")
        return None

    raw_dataset = load_from_disk(dataset_path)["validation"]

    reps_last = {}
    reps_avg4 = {}
    soft_labels = {}

    print("Extracting representations and soft labels per model...")
    for base_model, path in teacher_paths.items():
        print(f"  Processing {base_model}...")
        try:
            model, tokenizer = get_model_and_tokenizer(
                base_model, task_name, path, device
            )
            r_last, r_avg4, logits = get_representations_and_logits(
                model,
                tokenizer,
                raw_dataset,
                device,
                is_single_sentence=is_single_sentence,
            )
            reps_last[base_model] = r_last
            reps_avg4[base_model] = r_avg4
            # Use untempered distributions for model-level similarity.
            soft_labels[base_model] = F.softmax(logits, dim=-1)

            del model, tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    Failed for {base_model}: {e}")

    names = list(reps_last.keys())
    n = len(names)
    if n < 2:
        print("Not enough successful extractions. Skipping.")
        return None

    # Compute pairwise representation and output-space similarities.
    cka_last = np.zeros((n, n))
    cka_avg4 = np.zeros((n, n))
    jsd_mat = np.zeros((n, n))       # raw JSD (nats), T=1
    jsd_norm_mat = np.zeros((n, n))  # JSD / log(2), in [0, 1]

    for i in range(n):
        for j in range(n):
            cka_last[i, j] = cka(reps_last[names[i]], reps_last[names[j]])
            cka_avg4[i, j] = cka(reps_avg4[names[i]], reps_avg4[names[j]])
            jsd_mat[i, j] = jensen_shannon_divergence(
                soft_labels[names[i]], soft_labels[names[j]]
            )
            jsd_norm_mat[i, j] = jsd_mat[i, j] / np.log(2.0)

    # Report pairwise similarities.
    print("\n" + "-" * 86)
    print(f"PROXIMITY TABLE ({task_name.upper()} - {dataset_size_str})")
    print("-" * 86)
    print(
        f"{'Model Pair':<30} | {'CKA(last)':<9} | {'CKA(avg4)':<9} | "
        f"{'JSD(nats)':<9} | {'JSD norm':<9}"
    )
    print("-" * 86)
    for i in range(n):
        for j in range(i + 1, n):
            pair_name = f"{names[i]} <-> {names[j]}"
            print(
                f"{pair_name:<30} | {cka_last[i,j]:.3f}     | {cka_avg4[i,j]:.3f}     | "
                f"{jsd_mat[i,j]:.3f}     | {jsd_norm_mat[i,j]:.3f}"
            )

    # Prepare the output directory.
    out_dir = Path(Config.PROJECT_ROOT) / "results" / "proximity"
    out_dir.mkdir(exist_ok=True, parents=True)
    file_suffix = f"{task_name}_{dataset_size_str}_seed{seed_str}"

    # Visualize CKA and normalized JSD in three panels.
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    sns.heatmap(
        cka_last, annot=True, xticklabels=names, yticklabels=names,
        ax=axes[0], cmap="viridis", fmt=".2f", vmin=0, vmax=1,
    )
    axes[0].set_title("CKA (Last Layer) — primary")

    sns.heatmap(
        cka_avg4, annot=True, xticklabels=names, yticklabels=names,
        ax=axes[1], cmap="viridis", fmt=".2f", vmin=0, vmax=1,
    )
    axes[1].set_title("CKA (Avg Last 4 Layers) — robustness")

    sns.heatmap(
        jsd_norm_mat, annot=True, xticklabels=names, yticklabels=names,
        ax=axes[2], cmap="Reds", fmt=".2f", vmin=0, vmax=1,
    )
    axes[2].set_title("JSD (T=1.0, normalized)")

    plt.tight_layout()
    heatmaps_path = out_dir / f"proximity_heatmaps_{file_suffix}.png"
    plt.savefig(heatmaps_path, dpi=300)
    plt.close()

    # Construct a dendrogram from one minus final-layer CKA.
    dendrogram_path = out_dir / f"model_family_dendrogram_{file_suffix}.png"
    build_dendrogram(
        cka_last,
        names,
        f"Model Family Tree ({task_name.upper()} - {dataset_size_str})",
        dendrogram_path,
    )

    print(f"\nSaved heatmaps to {heatmaps_path}")
    print(f"Saved dendrogram to {dendrogram_path}")

    return names, cka_last, cka_avg4, jsd_mat, jsd_norm_mat


# Main entry point.
def main():
    device = Config.DEVICE

    models_dir = Path(Config.PROJECT_ROOT) / "models"
    base_models = [
        "bert",
        "electra-base",
        "roberta",
        "gpt2",
        "gpt-neo-125M",
        "opt-125m",
    ]

    print("Scanning for all teacher models...")
    groups = {}

    for base_model in base_models:
        search_path = models_dir / base_model
        if not search_path.exists():
            continue

        if base_model == "bert":
            for path in search_path.glob("teachers/teacher_*.pt"):
                filename = path.stem.replace("teacher_", "")
                parts_f = filename.split("_")
                task = parts_f[0]
                if len(parts_f) == 1:
                    size_str = "full"
                else:
                    if parts_f[-1] == "stratified":
                        size_str = parts_f[1]
                    else:
                        size_str = parts_f[1] + "_random"
                seed_str = "42"
                group_key = (task, size_str, seed_str)
                groups.setdefault(group_key, {})[base_model] = path
        else:
            for path in search_path.glob("**/baselines/teacher.pt"):
                parts = path.parts
                try:
                    base_idx = parts.index(base_model)
                    task = parts[base_idx + 1]
                    size_str = parts[base_idx + 2]
                    seed_str = parts[base_idx + 3].replace("seed_", "")
                    group_key = (task, size_str, seed_str)
                    groups.setdefault(group_key, {})[base_model] = path
                except ValueError:
                    continue

    valid_groups = {k: v for k, v in groups.items() if len(v) == len(base_models)}
    print(f"Found {len(valid_groups)} groups with ALL {len(base_models)} teachers.")

    # Global accumulators
    n_models = len(base_models)
    global_cka_last_sum = np.zeros((n_models, n_models))
    global_cka_avg4_sum = np.zeros((n_models, n_models))
    global_jsd_sum = np.zeros((n_models, n_models))
    global_jsd_norm_sum = np.zeros((n_models, n_models))
    global_counts = np.zeros((n_models, n_models))

    for (task, size_str, seed_str), teacher_paths in valid_groups.items():
        res = run_analysis_for_group(task, size_str, seed_str, teacher_paths, device)
        if res is None:
            continue
        names, cka_last, cka_avg4, jsd_mat, jsd_norm_mat = res
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                idx_i = base_models.index(name_i)
                idx_j = base_models.index(name_j)
                global_cka_last_sum[idx_i, idx_j] += cka_last[i, j]
                global_cka_avg4_sum[idx_i, idx_j] += cka_avg4[i, j]
                global_jsd_sum[idx_i, idx_j] += jsd_mat[i, j]
                global_jsd_norm_sum[idx_i, idx_j] += jsd_norm_mat[i, j]
                global_counts[idx_i, idx_j] += 1

    print("\n" + "=" * 70)
    print("  COMPUTING GLOBAL AVERAGE ACROSS ALL CONFIGS")
    print("=" * 70)

    valid_indices = [i for i in range(n_models) if global_counts[i, i] > 0]
    if not valid_indices:
        print("No valid models for global average.")
        return

    final_names = [base_models[i] for i in valid_indices]
    m = len(valid_indices)

    avg_cka_last = np.zeros((m, m))
    avg_cka_avg4 = np.zeros((m, m))
    avg_jsd = np.zeros((m, m))
    avg_jsd_norm = np.zeros((m, m))

    for x, i in enumerate(valid_indices):
        for y, j in enumerate(valid_indices):
            if global_counts[i, j] > 0:
                avg_cka_last[x, y] = global_cka_last_sum[i, j] / global_counts[i, j]
                avg_cka_avg4[x, y] = global_cka_avg4_sum[i, j] / global_counts[i, j]
                avg_jsd[x, y] = global_jsd_sum[i, j] / global_counts[i, j]
                avg_jsd_norm[x, y] = global_jsd_norm_sum[i, j] / global_counts[i, j]

    # Print global table
    print("\nGLOBAL AVERAGE PROXIMITY TABLE")
    print("-" * 86)
    print(
        f"{'Model Pair':<30} | {'CKA(last)':<9} | {'CKA(avg4)':<9} | "
        f"{'JSD(nats)':<9} | {'JSD norm':<9}"
    )
    print("-" * 86)
    for i in range(m):
        for j in range(i + 1, m):
            pair_name = f"{final_names[i]} <-> {final_names[j]}"
            print(
                f"{pair_name:<30} | {avg_cka_last[i,j]:.3f}     | "
                f"{avg_cka_avg4[i,j]:.3f}     | {avg_jsd[i,j]:.3f}     | "
                f"{avg_jsd_norm[i,j]:.3f}"
            )

    # Save global heatmaps — 3 panels
    out_dir = Path(Config.PROJECT_ROOT) / "results" / "proximity"
    out_dir.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    sns.heatmap(
        avg_cka_last, annot=True, xticklabels=final_names, yticklabels=final_names,
        ax=axes[0], cmap="viridis", fmt=".2f", vmin=0, vmax=1,
    )
    axes[0].set_title("Global Avg CKA (Last Layer) — primary")

    sns.heatmap(
        avg_cka_avg4, annot=True, xticklabels=final_names, yticklabels=final_names,
        ax=axes[1], cmap="viridis", fmt=".2f", vmin=0, vmax=1,
    )
    axes[1].set_title("Global Avg CKA (Avg Last 4 Layers) — robustness")

    sns.heatmap(
        avg_jsd_norm, annot=True, xticklabels=final_names, yticklabels=final_names,
        ax=axes[2], cmap="Reds", fmt=".2f", vmin=0, vmax=1,
    )
    axes[2].set_title("Global Avg JSD (T=1.0, normalized)")

    plt.tight_layout()
    heatmaps_path = out_dir / "proximity_heatmaps_GLOBAL_AVERAGE.png"
    plt.savefig(heatmaps_path, dpi=300)
    plt.close()

    # Global dendrogram
    dendrogram_path = out_dir / "model_family_dendrogram_GLOBAL_AVERAGE.png"
    build_dendrogram(
        avg_cka_last,
        final_names,
        "Global Average Model Family Tree (1 - CKA, last layer)",
        dendrogram_path,
    )

    print(f"\nSaved GLOBAL AVERAGE heatmaps to {heatmaps_path}")
    print(f"Saved GLOBAL AVERAGE dendrogram to {dendrogram_path}")
    print("\nAll Analysis Complete!")


if __name__ == "__main__":
    main()