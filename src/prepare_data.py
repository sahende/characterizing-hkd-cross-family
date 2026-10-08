"""Prepare GLUE data for knowledge-distillation experiments.

Stratified subsampling and train--validation splitting preserve class
distributions across dataset scales.
"""

import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split

try:
    from transformers import GPT2Tokenizer
except ImportError:  # pragma: no cover
    GPT2Tokenizer = None


def get_tokenizer_for_model():
    tokenizer = AutoTokenizer.from_pretrained(
        Config.TEACHER_MODEL,
        local_files_only=Config.local_files_only()
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"Tokenizer for {Config.MODEL_NAME} has no pad or eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


def prepare_single_task(task_name, tokenizer, max_train_samples=None, stratified=True):
    """
    Prepare DataLoaders for a single GLUE task.

    Args:
        max_train_samples: If set, subsample training data to this size.
        stratified: If True (default), both subsampling and train/val split
                    are stratified to preserve class distributions.
                    If False, uses random (non-stratified) sampling.
    """
    print(f"  Loading {task_name.upper()}...")

    # Load from the Hugging Face cache and reuse the local copy offline.
    import datasets as _datasets
    import socket

    def _has_internet(host="8.8.8.8", port=53, timeout=3):
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except OSError:
            return False

    previous_offline = os.environ.get("HF_DATASETS_OFFLINE")
    if Config.DATA_OFFLINE or not _has_internet():
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print(f"    [offline] Loading {task_name.upper()} from cache...")
    else:
        os.environ.pop("HF_DATASETS_OFFLINE", None)

    try:
        dataset = load_from_disk(os.path.join(Config.DATA_DIR, task_name))
    finally:
        if previous_offline is None:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
        else:
            os.environ["HF_DATASETS_OFFLINE"] = previous_offline

    original_size = len(dataset['train'])
    print(f"    Original train size: {original_size}")

    # Step 1: subsample the training split, stratified by default.
    if max_train_samples is not None and original_size > max_train_samples:
        if stratified:
            labels = np.array(dataset['train']['label'])
            indices = np.arange(original_size)
            strat_indices, _ = train_test_split(
                indices,
                train_size=max_train_samples,
                stratify=labels,
                random_state=Config.SEED
            )
            dataset['train'] = dataset['train'].select(strat_indices.tolist())

            sampled_labels = np.array(dataset['train']['label'])
            unique, counts = np.unique(sampled_labels, return_counts=True)
            dist_str = ", ".join([
                f"class {u}: {c} ({c/len(sampled_labels)*100:.1f}%)"
                for u, c in zip(unique, counts)
            ])
            print(f"    ⚠ Stratified subsample: {original_size} → {max_train_samples}")
            print(f"       Distribution: {dist_str}")
        else:
            # Apply random subsampling without changing the later split procedure.
            rng = np.random.RandomState(Config.SEED)
            indices = rng.choice(original_size, max_train_samples, replace=False)
            dataset['train'] = dataset['train'].select(indices.tolist())
            print(f"    ⚠ Random subsample: {original_size} → {max_train_samples}")
    else:
        print(f"    Using full train: {original_size} samples")

    # Tokenize the task-specific text fields.
    is_single_sentence = task_name in ["sst2", "cola"]

    def preprocess_function(examples):
        if is_single_sentence:
            return tokenizer(
                examples['sentence'],
                truncation=True,
                padding='max_length',
                max_length=Config.MAX_SEQ_LENGTH
            )
        else:
            return tokenizer(
                examples['sentence1'],
                examples['sentence2'],
                truncation=True,
                padding='max_length',
                max_length=Config.MAX_SEQ_LENGTH
            )

    tokenized = dataset.map(preprocess_function, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    # Include token type IDs only when produced by the tokenizer.
    format_columns = ["input_ids", "attention_mask"]
    if "token_type_ids" in tokenized['train'].features:
        format_columns.append("token_type_ids")
    format_columns.append("labels")

    tokenized.set_format(type="torch", columns=format_columns)

    # Step 2: split the training data, stratified by default.
    train_data = tokenized['train']
    train_indices = np.arange(len(train_data))
    train_labels = np.array(train_data['labels'])

    train_size = int((1 - Config.VAL_SPLIT) * len(train_data))

    if stratified:
        # Preserve the label distribution in both partitions.
        train_idx, val_idx = train_test_split(
            train_indices,
            train_size=train_size,
            stratify=train_labels,
            random_state=Config.SEED
        )
        train_subset = Subset(train_data, train_idx)
        val_subset = Subset(train_data, val_idx)
        
        # Report the label distributions for reproducibility.
        train_dist = np.unique(train_labels[train_idx], return_counts=True)
        val_dist = np.unique(train_labels[val_idx], return_counts=True)
        print(f"    Train: {len(train_idx)} samples | "
              + " | ".join([f"class {u}: {c} ({c/len(train_idx)*100:.1f}%)"
                            for u, c in zip(train_dist[0], train_dist[1])]))
        print(f"    Val:   {len(val_idx)} samples   | "
              + " | ".join([f"class {u}: {c} ({c/len(val_idx)*100:.1f}%)"
                            for u, c in zip(val_dist[0], val_dist[1])]))
    else:
        # Use a random split when stratification is disabled.
        train_dataset, val_dataset = random_split(
            train_data,
            [train_size, len(train_data) - train_size],
            generator=torch.Generator().manual_seed(Config.SEED)
        )
        # ``random_split`` already returns ``Subset`` instances.
        train_subset = train_dataset
        val_subset = val_dataset
        
        # Report the resulting partition sizes.
        print(f"    Train: {len(train_subset)}, Val: {len(val_subset)}")

    return {
        'train': DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True),
        'val':   DataLoader(val_subset,   batch_size=Config.BATCH_SIZE, shuffle=False),
        'test':  DataLoader(
            tokenized['validation'],
            batch_size=Config.BATCH_SIZE,
            shuffle=False
        ),
        'train_dataset': train_subset,
        'val_dataset':   val_subset,
        'train_size':    len(train_subset)
    }


def prepare_all_tasks(task_list=None, task_subsample_sizes=None, stratified=None):
    """
    Prepare data for given task list.

    Args:
        task_subsample_sizes: Dict mapping task_name → max_train_samples.
                              Defaults to Config.DATASET_SIZES.
        stratified: Passed through to prepare_single_task. If omitted, uses
                Config.STRATIFIED.
    """
    if task_list is None:
        task_list = Config.TASKS
    if stratified is None:
        stratified = Config.STRATIFIED

    if task_subsample_sizes is None:
        task_subsample_sizes = Config.DATASET_SIZES

    tokenizer = get_tokenizer_for_model()
    if Config.is_gpt_model():
        print(f"  GPT tokenizer: pad_token set to {tokenizer.pad_token}")

    print(f"Preparing tasks: {task_list}")
    if task_subsample_sizes:
        print(f"  Subsampling config: {task_subsample_sizes}")

    task_dataloaders = {}
    for task in task_list:
        max_samples = task_subsample_sizes.get(task, None)
        task_dataloaders[task] = prepare_single_task(
            task, tokenizer,
            max_train_samples=max_samples,
            stratified=stratified
        )

    return task_dataloaders, tokenizer


def prepare_combined_tasks(task_list, task_subsample_sizes=None, stratified=None):
    """Combine multiple tasks into one DataLoader."""
    if stratified is None:
        stratified = Config.STRATIFIED
    if task_subsample_sizes is None:
        task_subsample_sizes = Config.DATASET_SIZES

    tokenizer = get_tokenizer_for_model()
    print(f"Preparing COMBINED tasks: {task_list}")

    task_data = {}
    for task in task_list:
        max_samples = task_subsample_sizes.get(task, None)
        task_data[task] = prepare_single_task(
            task, tokenizer,
            max_train_samples=max_samples,
            stratified=stratified
        )

    train_datasets = [task_data[t]['train_dataset'] for t in task_list]
    combined_train = ConcatDataset(train_datasets)

    val_datasets = [task_data[t]['val_dataset'] for t in task_list]
    combined_val = ConcatDataset(val_datasets)

    combined_loaders = {
        'train': DataLoader(combined_train, batch_size=Config.BATCH_SIZE, shuffle=True),
        'val':   DataLoader(combined_val,   batch_size=Config.BATCH_SIZE, shuffle=False),
    }

    per_task_test = {t: task_data[t]['test'] for t in task_list}

    print(f"  Combined train: {len(combined_train)}, val: {len(combined_val)}")

    return combined_loaders, per_task_test, tokenizer


if __name__ == "__main__":
    print("=" * 60)
    print("  GLUE Data Preparation (STRATIFIED)")
    print("=" * 60)

    task_dataloaders, _ = prepare_all_tasks(
        task_list=["rte", "mrpc", "cola", "sst2"],
        task_subsample_sizes=Config.DATASET_SIZES
    )
    print("\n✓ Data preparation complete!")