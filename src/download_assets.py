"""Download the model and GLUE datasets used by the experiments."""

import os

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import Config


def download_assets():
    os.makedirs(os.path.dirname(Config.TEACHER_MODEL), exist_ok=True)
    os.makedirs(Config.DATA_DIR, exist_ok=True)

    print(f"Downloading model and tokenizer: {Config.MODEL_SOURCE}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_SOURCE)
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_SOURCE,
        num_labels=2,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    tokenizer.save_pretrained(Config.TEACHER_MODEL)
    model.save_pretrained(Config.TEACHER_MODEL)

    for task in Config.TASKS:
        task_path = os.path.join(Config.DATA_DIR, task)
        if os.path.exists(task_path):
            print(f"  {task.upper()} already exists: {task_path}")
            continue
        print(f"Downloading GLUE/{task}...")
        dataset = load_dataset("glue", task)
        dataset.save_to_disk(task_path)

    print("Local assets are ready.")
    print(f"  Model: {Config.TEACHER_MODEL}")
    print(f"  Data:  {Config.DATA_DIR}")


if __name__ == "__main__":
    download_assets()