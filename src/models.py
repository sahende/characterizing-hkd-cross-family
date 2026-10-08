"""Model definitions for hierarchical knowledge distillation."""

import os
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)
from config import Config


def _has_internet(host="8.8.8.8", port=53, timeout=3):
    """Return True if we can reach the internet."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


# Enable Hugging Face offline mode when no network connection is available.
if not _has_internet():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    print("[models] No internet detected — running in offline mode (using local cache).")
else:
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)


def get_teacher_model(task_name, num_labels=2):
    print(f"Loading teacher model: {Config.TEACHER_MODEL} for {task_name}...")

    model = AutoModelForSequenceClassification.from_pretrained(
        Config.TEACHER_MODEL,
        num_labels=num_labels,
        local_files_only=Config.local_files_only()
    )
    if model.config.pad_token_id is None and model.config.eos_token_id is not None:
        model.config.pad_token_id = model.config.eos_token_id
    teacher_layers = get_num_layers(model.config)
    if teacher_layers != Config.EXPECTED_TEACHER_LAYERS:
        raise ValueError(
            f"Teacher {Config.MODEL_NAME} must have "
            f"{Config.EXPECTED_TEACHER_LAYERS} layers, found {teacher_layers}."
        )

    print(f"  Teacher parameters: {count_parameters(model):,}")
    return model


def get_student_model(num_labels=2):
    print(f"Creating student model ({Config.STUDENT_NUM_LAYERS} layers, random init)...")

    config = AutoConfig.from_pretrained(
        Config.TEACHER_MODEL,
        local_files_only=Config.local_files_only()
    )
    config.num_labels = num_labels
    _set_num_layers(config, Config.STUDENT_NUM_LAYERS)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = getattr(config, "eos_token_id", None)
    model = AutoModelForSequenceClassification.from_config(config)

    model.init_weights()
    print(f"  Student parameters: {count_parameters(model):,}")
    return model


def _set_num_layers(config, num_layers):
    """Set the layer-count field used by common encoder and decoder configs."""
    for field_name in ("num_hidden_layers", "num_layers", "n_layer"):
        if hasattr(config, field_name):
            setattr(config, field_name, num_layers)
            return
    raise ValueError(f"Unsupported model config without a layer-count field: {config.model_type}")


def get_num_layers(config):
    """Read the layer-count field used by common encoder and decoder configs."""
    for field_name in ("num_hidden_layers", "num_layers", "n_layer"):
        if hasattr(config, field_name):
            return getattr(config, field_name)
    raise ValueError(f"Unsupported model config without a layer-count field: {config.model_type}")


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DistillationLoss(nn.Module):
    """
    Compute the logit-based distillation objective of Hinton et al. (2015).

    The objective combines hard-label supervision with a temperature-scaled
    Kullback--Leibler divergence between teacher and student predictions:

        L = alpha * CE(student_logits, labels)
            + (1 - alpha) * T^2 * KL(teacher || student).
    """
    
    def __init__(self, temperature=Config.TEMPERATURE, alpha=Config.ALPHA):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        """Return the total, hard-label, and soft-label loss components."""
        # Hard-label supervision uses cross-entropy with the ground-truth labels.
        ce_loss = F.cross_entropy(student_logits, labels)

        # Soft-label supervision uses temperature-scaled teacher predictions.
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

        kl_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combine hard-label and soft-label supervision.
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        return total_loss, ce_loss, kl_loss