"""Analyze teacher entropy, calibration, and predictive performance."""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

from config import Config
from models import get_teacher_model
from prepare_data import prepare_all_tasks


# Metric computation.
def compute_softened_entropy(logits, temperature=Config.TEMPERATURE, eps=1e-12):
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = torch.log(probs + eps)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def compute_confidence(logits):
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values


def compute_calibration_metrics(confidences, accuracies, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if bin_lower == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


# Teacher analysis.
def analyze_teacher(teacher, dataloader, device, task_name):
    """Analyze teacher model."""
    teacher.eval()
    
    all_teacher_entropy = []
    all_teacher_conf = []
    all_teacher_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  {task_name}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"]
            
            labels = batch["labels"]
            all_labels.extend(labels.cpu().numpy())
            
            # Compute teacher predictions and uncertainty measures.
            t_out = teacher(**inputs)
            all_teacher_entropy.append(compute_softened_entropy(t_out.logits))
            all_teacher_conf.append(compute_confidence(t_out.logits))
            all_teacher_preds.append(torch.argmax(t_out.logits, dim=-1))
    
    all_labels_tensor = torch.tensor(all_labels, device=device)
    
    # Aggregate teacher-level statistics.
    t_entropy = torch.cat(all_teacher_entropy)
    t_conf = torch.cat(all_teacher_conf)
    t_preds = torch.cat(all_teacher_preds)
    t_acc = (t_preds == all_labels_tensor).float()
    t_mcc = matthews_corrcoef(all_labels_tensor.cpu().numpy(), t_preds.cpu().numpy())
    
    teacher_summary = {
        'mean_softened_entropy': float(t_entropy.mean()),
        'accuracy': float(t_acc.mean()),
        'mcc': float(t_mcc),
        'calibration': {'ece': compute_calibration_metrics(t_conf.cpu().numpy(), t_acc.cpu().numpy())}
    }
    
    return teacher_summary


# Model loading.
def load_teacher(path, device):
    """Load teacher model directly from path."""
    import re
    # Recover the task name from the artifact path.
    task_name = Path(path).parts[-5]  # The fifth component from the end is the task.
    teacher = get_teacher_model(task_name, num_labels=2)
    
    if os.path.exists(path):
        teacher.load_state_dict(torch.load(path, map_location=device))
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"    ✓ Teacher: {path}")
        return teacher
    print(f"    ✗ Teacher not found at {path}")
    return None


def run_config(task_name, variant_label, path, device, all_results, test_loaders):
    """Run analysis for one configuration."""
    print(f"\n{'─'*60}")
    print(f"  {task_name.upper()} - {variant_label}")
    print(f"{'─'*60}")
    
    if task_name not in test_loaders:
        # Only the test split is required for this analysis.
        target_data, _ = prepare_all_tasks([task_name], {})
        test_loaders[task_name] = target_data[task_name]['test']
    
    test_loader = test_loaders[task_name]
    
    teacher = load_teacher(path, device)
    if teacher is None:
        return
    
    summary = analyze_teacher(teacher, test_loader, device, task_name)
    
    key = f"{task_name}_{variant_label}"
    all_results[key] = summary
    
    # Report the configuration summary.
    print(f"    {'Teacher':<25} Acc={summary['accuracy']:.4f} MCC={summary['mcc']:.4f} "
          f"Entropy={summary['mean_softened_entropy']:.4f} ECE={summary['calibration']['ece']:.4f}")
    
    del teacher
    torch.cuda.empty_cache()


def main():
    device = Config.DEVICE
    
    print("\n" + "=" * 70)
    print("  TEACHER ENTROPY & CALIBRATION ANALYSIS (Auto-Scan)")
    print("=" * 70)
    
    all_results = {}
    test_loaders = {}
    
    # Discover all available teacher checkpoints.
    base_model = os.path.basename(Config.TEACHER_MODEL)
    search_path = os.path.join(Config.MODEL_SAVE_PATH, base_model, "**", "teacher.pt")
    from pathlib import Path
    import glob
    teacher_files = glob.glob(search_path, recursive=True)
    
    configs = []
    for path in teacher_files:
        try:
            parts = Path(path).parts
            base_idx = parts.index(base_model)
            task = parts[base_idx + 1]
            dataset_size_str = parts[base_idx + 2]
            seed_str = parts[base_idx + 3]
            
            variant_label = f"{dataset_size_str}_{seed_str}"
            configs.append((task, variant_label, path))
        except ValueError:
            continue
    
    print(f"Found {len(configs)} teacher models to evaluate.")
    for task, variant, path in configs:
        run_config(task, variant, path, device, all_results, test_loaders)
    
    # Save JSON-serializable results.
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert_to_serializable(v) for v in obj]
        return obj
    
    output_path = Config.get_results_path("teacher_entropy_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    print(f"\n✓ JSON saved to: {output_path}")


if __name__ == "__main__":
    main()
