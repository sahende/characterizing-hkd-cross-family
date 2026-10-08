"""Evaluate depth--performance trade-offs under hierarchical distillation.

The experiment uses multi-seed depth ablations, confidence intervals, paired
statistical tests, effect sizes, quadratic regression, and efficiency metrics.
Distillation transfers logits only through cross-entropy and KL-divergence
terms; hidden-state and attention transfer are not used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import json
import time
import numpy as np
from scipy.stats import ttest_rel, sem, t
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoConfig, AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from config import Config
from models import get_teacher_model, get_student_model, DistillationLoss, count_parameters
from prepare_data import prepare_all_tasks


# Reproducibility configuration.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Experimental seeds and assistant depths.
SEEDS = [42,52,62,72,82]
ALL_DEPTHS = [1, 2, 4, 6, 8, 10]


# Assistant model construction.
def get_assistant_model(num_labels=2, num_layers=8):
    """Create an assistant model matching the local teacher architecture."""
    config = AutoConfig.from_pretrained(
        Config.TEACHER_MODEL,
        local_files_only=Config.local_files_only()
    )
    config.num_labels = num_labels
    for field_name in ("num_hidden_layers", "num_layers", "n_layer"):
        if hasattr(config, field_name):
            setattr(config, field_name, num_layers)
            break
    else:
        raise ValueError(f"Unsupported model config without a layer-count field: {config.model_type}")
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = getattr(config, "eos_token_id", None)
    model = AutoModelForSequenceClassification.from_config(config)
    
    model.init_weights()
    return model


# 95% confidence interval.
def compute_95ci(data):
    n = len(data)
    if n < 2:
        return {'mean': round(np.mean(data), 4), 'lower': float('nan'), 'upper': float('nan'), 'ci_width': float('nan')}
    mean = np.mean(data)
    std_err = sem(data)
    ci = t.ppf(0.975, n-1) * std_err
    return {
        'mean': round(mean, 4),
        'lower': round(mean - ci, 4),
        'upper': round(mean + ci, 4),
        'ci_width': round(ci, 4)
    }


# Quadratic regression.
def fit_quadratic(depths, f1s):
    depths = np.array(depths)
    f1s = np.array(f1s)
    
    coeffs = np.polyfit(depths, f1s, 2)
    a, b, c = coeffs
    
    predicted = np.polyval(coeffs, depths)
    
    ss_res = np.sum((f1s - predicted) ** 2)
    ss_tot = np.sum((f1s - np.mean(f1s)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    if a != 0:
        optimal_depth = -b / (2 * a)
    else:
        optimal_depth = depths[np.argmax(f1s)]
    
    return {
        'a': round(float(a), 6),
        'b': round(float(b), 6),
        'c': round(float(c), 6),
        'r_squared': round(float(r_squared), 4),
        'optimal_depth': round(float(optimal_depth), 2),
        'equation': f"F1 = {a:.6f}*D² + {b:.6f}*D + {c:.6f}",
        'predicted': [round(float(p), 4) for p in predicted]
    }


# Paired Cohen's d.
def cohens_d_paired(x, y):
    diff = np.array(x) - np.array(y)
    std_diff = np.std(diff, ddof=1)
    return np.mean(diff) / std_diff if std_diff > 0 else 0


def interpret_effect_size(d):
    d_abs = abs(d)
    if d_abs > 0.8: return 'large'
    elif d_abs > 0.5: return 'medium'
    elif d_abs > 0.2: return 'small'
    else: return 'negligible'


# Evaluation metrics.
def compute_all_metrics(preds, labels):
    cm = confusion_matrix(labels, preds)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_per_class": f1_score(labels, preds, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(labels, preds, average=None, zero_division=0).tolist(),
        "confusion_matrix": cm.tolist()
    }


def compute_kl_divergence(t_logits, s_logits, T=4.0):
    t_prob = F.softmax(t_logits / T, dim=-1)
    s_logprob = F.log_softmax(s_logits / T, dim=-1)
    return F.kl_div(s_logprob, t_prob, reduction="batchmean").item()


def compute_alignment_gap(t_logits, s_logits):
    return F.mse_loss(s_logits, t_logits).item()


def evaluate(model, loader, device):
    """Evaluate a model using only the inputs supported by its tokenizer."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in loader:
            # Move the batch to the selected device.
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Pass token type IDs only when generated by the tokenizer.
            inputs = {
                "input_ids": batch["input_ids"], 
                "attention_mask": batch["attention_mask"]
            }
            # Retain compatibility with tokenizers that omit token type IDs.
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"]
            
            outputs = model(**inputs)
            loss = criterion(outputs.logits, batch["labels"])
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    metrics = compute_all_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics["loss"], metrics


# Checkpoint loading.
def load_model_if_exists(model, save_path, device):
    """Load model from path if exists. Returns (model, loaded_flag)."""
    if os.path.exists(save_path):
        try:
            state_dict = torch.load(save_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            return model, True
        except Exception as e:
            print(f"    ⚠ Failed to load {save_path}: {e}")
            return model, False
    return model, False


# Knowledge-distillation training.
def train_kd_epoch(teacher, student, loader, optimizer, scheduler, loss_fn, device, epoch, mode="KD"):
    """Train one epoch with logits-only knowledge distillation."""
    teacher.eval()
    student.train()
    total_loss = 0
    total_kl = 0
    total_gap = 0
    grad_norms = []
    n_batches = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc=f"{mode} E{epoch+1}"):
        # Move the batch to the selected device.
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Pass token type IDs only when generated by the tokenizer.
        inputs = {
            "input_ids": batch["input_ids"], 
            "attention_mask": batch["attention_mask"]
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"]

        # Compute teacher logits without gradient tracking.
        with torch.no_grad():
            t_out = teacher(**inputs)

        # Compute student logits.
        s_out = student(**inputs)
        loss, ce_loss, kl_loss = loss_fn(s_out.logits, t_out.logits, batch["labels"])

        # Backpropagate the distillation objective.
        optimizer.zero_grad()
        loss.backward()
        
        # Track the un-clipped gradient norm.
        total_norm = 0
        for p in student.parameters():
            if p.grad is not None: 
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms.append(total_norm ** 0.5)
        
        # Clip gradients to stabilize optimization.
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Accumulate optimization and alignment statistics.
        total_loss += loss.item()
        total_kl += compute_kl_divergence(t_out.logits, s_out.logits, Config.TEMPERATURE)
        total_gap += compute_alignment_gap(t_out.logits, s_out.logits)
        n_batches += 1
        
        # Accumulate student predictions and labels.
        all_preds.extend(torch.argmax(s_out.logits, -1).cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    metrics = compute_all_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / n_batches
    metrics["kl_divergence"] = total_kl / n_batches
    metrics["alignment_gap"] = total_gap / n_batches
    metrics["gradient_norm"] = np.mean(grad_norms)
    metrics["gradient_std"] = np.std(grad_norms)
    return metrics


def train_kd_loop(teacher, student, train_loader, val_loader, test_loader, device, 
                  epochs, lr_mult=1.0, mode="KD", save_path=None, force_retrain=False):
    """Train KD loop with checkpoint resume capability."""
    
    # Try to load existing model
    if not force_retrain and save_path:
        student, loaded = load_model_if_exists(student, save_path, device)
        if loaded:
            print(f"    ✓ Loaded from checkpoint, skipping training")
            student.eval()
            _, test_metrics = evaluate(student, test_loader, device)
            results = {
                'test': test_metrics,
                'train': [],
                'val': [],
                'training_time_sec': 0,
                'training_time_min': 0,
                'loaded_from_checkpoint': True
            }
            return student, results
    
    # Train from scratch
    teacher.eval()
    for p in teacher.parameters(): 
        p.requires_grad = False

    student.to(device)
    torch.cuda.empty_cache()
    
    loss_fn = DistillationLoss(Config.TEMPERATURE, Config.ALPHA)
    optimizer = AdamW(
        student.parameters(), 
        lr=Config.STUDENT_LR * lr_mult, 
        weight_decay=Config.WEIGHT_DECAY
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        int(total_steps * Config.WARMUP_RATIO), 
        total_steps
    )

    torch.cuda.reset_peak_memory_stats(device)
    training_start = time.time()
    
    best_val_f1 = 0
    best_state = None
    results = {'train': [], 'val': [], 'test': None}
    val_f1s = []

    for epoch in range(epochs):
        train_metrics = train_kd_epoch(
            teacher, student, train_loader, optimizer, scheduler, loss_fn, device, epoch, mode)
        results['train'].append({'epoch': epoch+1, **train_metrics})

        _, val_metrics = evaluate(student, val_loader, device)
        results['val'].append({'epoch': epoch+1, **val_metrics})
        val_f1s.append(val_metrics['f1_macro'])

        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_state = copy.deepcopy(student.state_dict())

    training_time = time.time() - training_start

    if best_state: 
        student.load_state_dict(best_state)
        if save_path:
            torch.save(best_state, save_path)
            print(f"    ✓ Model saved to: {save_path}")

    _, test_metrics = evaluate(student, test_loader, device)
    results['test'] = test_metrics
    results['training_time_sec'] = round(training_time, 1)
    results['training_time_min'] = round(training_time / 60, 2)
    results['loaded_from_checkpoint'] = False
    
    return student, results


# JSON serialization helper.
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (bool, np.bool_)):
        return int(obj)
    else:
        return obj


# Main entry point.
def main():
    device = Config.DEVICE
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  M2 - HIERARCHICAL KD ({Config.MODEL_NAME}) - Depth-Performance Analysis")
    print("=" * 70)
    print(f"  Teacher: {Config.TEACHER_MODEL}")
    print(f"  Student layers: {Config.STUDENT_NUM_LAYERS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Assistant depths: {ALL_DEPTHS}")
    print("=" * 70)

    all_results = {}

    for task in Config.TARGET_TASKS:
        for size in Config.get_task_sizes(task):
            run_key = f"{task}_{size}"
            print(f"\n{'#'*60}")
            print(f"  TASK: {task.upper()}  |  SIZE: {size}")
            print(f"{'#'*60}")

            # Load data for the current task and dataset size.
            subsample = Config.size_to_subsample(task, size)
            target_data, _ = prepare_all_tasks(
                [task], subsample, stratified=Config.STRATIFIED
            )

            train_loader = target_data[task]['train']
            val_loader   = target_data[task]['val']
            test_loader  = target_data[task]['test']

            # Evaluate the teacher ceiling.
            print(f"\n  --- Loading Teacher Model ---")
            teacher_path = Config.get_model_path(task, "baselines", "teacher.pt", dataset_size=size)
            teacher = get_teacher_model(task, num_labels=2)
            
            if os.path.exists(teacher_path):
                teacher.load_state_dict(torch.load(teacher_path, map_location=device))
                print(f"  ✓ Loaded teacher from {teacher_path}")
            else:
                print(f"  ✗ Teacher not found at {teacher_path}")
                print(f"  ⚠ Run train_baseline.py first to train the teacher!")
                continue
                
            teacher.to(device).eval()
            for p in teacher.parameters(): 
                p.requires_grad = False
            
            _, teacher_test = evaluate(teacher, test_loader, device)
            teacher_f1 = teacher_test['f1_macro']
            print(f"  ✓ Teacher Ceiling: F1={teacher_f1:.4f}")

            # ================================================================
            # FAIR INIT (Same random init for all seeds)
            # ================================================================
            torch.manual_seed(Config.SEED)
            base_student_init = get_student_model(num_labels=2).state_dict()
            
            assistant_inits = {}
            for depth in ALL_DEPTHS:
                torch.manual_seed(Config.SEED)
                assistant_inits[depth] = get_assistant_model(num_labels=2, num_layers=depth).state_dict()

            task_results = {
                'teacher_ceiling': {'f1': teacher_f1},
                'direct_kd': {},
                'depth_ablation': {}
            }

            # Direct distillation from the teacher to the student.
            print(f"\n  --- DIRECT KD: Teacher (12L) → Student ({Config.STUDENT_NUM_LAYERS}L) [{len(SEEDS)} Seeds] ---")
            direct_seeds = []
            
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.empty_cache()
                
                student_direct = get_student_model(num_labels=2)
                student_direct.load_state_dict(copy.deepcopy(base_student_init))
                
                direct_save_path = Config.get_model_path(
                    task, "hierarchical_kd", "direct_kd.pt", seed=seed, dataset_size=size
                )
            
                student_direct, dr = train_kd_loop(
                    teacher, student_direct, train_loader, val_loader, test_loader,
                    device, Config.ADAPT_EPOCHS, lr_mult=1.0, 
                    mode=f"Direct KD (seed={seed})",
                    save_path=direct_save_path, force_retrain=False
                )
                direct_seeds.append(dr)
                del student_direct
                torch.cuda.empty_cache()

            direct_f1s = [r['test']['f1_macro'] for r in direct_seeds]
            direct_ci = compute_95ci(direct_f1s)
            direct_times = [r['training_time_sec'] for r in direct_seeds]
            
            task_results['direct_kd'] = {
                'f1_ci': direct_ci,
                'f1_std': round(np.std(direct_f1s), 4),
                'training_time_min_mean': round(np.mean(direct_times) / 60, 2)
            }
            print(f"  ✓ Direct KD: F1={direct_ci['mean']:.4f} [95%CI: {direct_ci['lower']:.4f}-{direct_ci['upper']:.4f}]")

            # Ablate assistant depth in the hierarchical distillation pipeline.
            for depth in ALL_DEPTHS:
                print(f"\n  --- HKD ({depth}L): Assistant ({depth}L) → Student ({Config.STUDENT_NUM_LAYERS}L) [{len(SEEDS)} Seeds] ---")
                depth_seeds = []
                
                for seed in SEEDS:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.cuda.empty_cache()
                    
                    # Stage 1: distill the teacher into the assistant.
                    assistant_save_path = Config.get_model_path(
                        task, "hierarchical_kd", f"assistant_depth{depth}L.pt", 
                        seed=seed, dataset_size=size
                    )
                    
                    assistant = get_assistant_model(num_labels=2, num_layers=depth)
                    assistant.load_state_dict(copy.deepcopy(assistant_inits[depth]))
                    
                    assistant, s1 = train_kd_loop(
                        teacher, assistant, train_loader, val_loader, test_loader,
                        device, Config.ADAPT_EPOCHS, lr_mult=1.0, 
                        mode=f"Stage1 {depth}L (seed={seed})",
                        save_path=assistant_save_path, force_retrain=False
                    )
                    
                    # Stage 2: distill the assistant into the student.
                    student_save_path = Config.get_model_path(
                        task, "hierarchical_kd", f"student_hkd_depth{depth}L.pt", 
                        seed=seed, dataset_size=size
                    )
                    
                    student_hkd = get_student_model(num_labels=2)
                    student_hkd.load_state_dict(copy.deepcopy(base_student_init))
                    
                    student_hkd, s2 = train_kd_loop(
                        assistant, student_hkd, train_loader, val_loader, test_loader,
                        device, Config.ADAPT_EPOCHS, lr_mult=0.5, 
                        mode=f"Stage2 {depth}L→{Config.STUDENT_NUM_LAYERS}L (seed={seed})",
                        save_path=student_save_path, force_retrain=False
                    )
                    
                    depth_seeds.append({'stage1': s1, 'stage2': s2, 'seed': seed})
                    
                    s1_loaded = s1.get('loaded_from_checkpoint', False)
                    s2_loaded = s2.get('loaded_from_checkpoint', False)
                    if s1_loaded or s2_loaded:
                        print(f"    ⚡ Seed {seed}: Stage1={'loaded' if s1_loaded else 'trained'}, "
                              f"Stage2={'loaded' if s2_loaded else 'trained'}")
                    
                    del assistant, student_hkd
                    torch.cuda.empty_cache()

                # Aggregate results for the current assistant depth.
                valid_seeds = [s for s in depth_seeds if s['stage2']['test'] is not None]
                
                if len(valid_seeds) < 2:
                    print(f"    ⚠ Only {len(valid_seeds)} valid seeds for depth {depth}L, using all available")
                    if len(valid_seeds) == 0:
                        continue
                
                s2_f1s = [s['stage2']['test']['f1_macro'] for s in valid_seeds]
                s2_ci = compute_95ci(s2_f1s)
                
                # Match seeds with direct KD for the paired statistical test.
                matching_direct_f1s = []
                for s in valid_seeds:
                    seed_idx = SEEDS.index(s['seed']) if s['seed'] in SEEDS else -1
                    if seed_idx >= 0 and seed_idx < len(direct_f1s):
                        matching_direct_f1s.append(direct_f1s[seed_idx])
                
                # Compute the paired t-test and paired Cohen's d.
                if len(matching_direct_f1s) >= 2 and len(s2_f1s) >= 2 and len(matching_direct_f1s) == len(s2_f1s):
                    _, p_val = ttest_rel(s2_f1s, matching_direct_f1s)
                    d_val = cohens_d_paired(s2_f1s, matching_direct_f1s)
                else:
                    p_val = float('nan')
                    d_val = float('nan')
                
                gain = round(s2_ci['mean'] - direct_ci['mean'], 4)
                s2_times = [s['stage1']['training_time_sec'] + s['stage2']['training_time_sec'] for s in valid_seeds]
                overhead = round(np.mean(s2_times) / np.mean(direct_times), 2) if np.mean(direct_times) > 0 else float('inf')

                task_results['depth_ablation'][str(depth)] = {
                    'f1_ci': s2_ci,
                    'f1_std': round(np.std(s2_f1s), 4),
                    'gain_vs_direct': gain,
                    'p_value': round(p_val, 4) if not np.isnan(p_val) else None,
                    'cohens_d_paired': round(d_val, 4) if not np.isnan(d_val) else None,
                    'effect_size': interpret_effect_size(d_val) if not np.isnan(d_val) else 'N/A',
                    'compute_overhead': overhead,
                    'num_valid_seeds': len(valid_seeds),
                    'num_total_seeds': len(SEEDS)
                }

                print(f"  ✓ {depth}L: F1={s2_ci['mean']:.4f} [95%CI: {s2_ci['lower']:.4f}-{s2_ci['upper']:.4f}], "
                      f"Δ={gain:+.4f}, p={p_val:.4f}, d={d_val:.4f}, seeds={len(valid_seeds)}/{len(SEEDS)}")

            # Fit a quadratic depth--performance relationship.
            valid_depths = []
            valid_f1s = []
            for d in ALL_DEPTHS:
                if str(d) in task_results['depth_ablation']:
                    valid_depths.append(d)
                    valid_f1s.append(task_results['depth_ablation'][str(d)]['f1_ci']['mean'])
            
            if len(valid_depths) >= 3:
                quad_fit = fit_quadratic(valid_depths, valid_f1s)
                task_results['quadratic_fit'] = quad_fit
                print(f"\n  --- Quadratic Fit ---")
                print(f"  Equation: {quad_fit['equation']}")
                print(f"  R² = {quad_fit['r_squared']:.4f}")
                print(f"  Optimal depth = {quad_fit['optimal_depth']:.2f}L")
            else:
                task_results['quadratic_fit'] = None
                print(f"\n  --- Quadratic Fit: SKIPPED (need ≥3 depths, have {len(valid_depths)}) ---")

            all_results[run_key] = task_results

            del teacher
            torch.cuda.empty_cache()

    # Serialize the complete experiment results.
    all_results = make_serializable(all_results)
    results_path = Config.get_results_path(f"m2_final_results_{Config.MODEL_NAME}.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    # Report the final depth--performance summary.
    print(f"\n{'='*120}")
    print(f"  M2 FINAL: DEPTH-PERFORMANCE RELATIONSHIP ({Config.MODEL_NAME})")
    print(f"{'='*120}")
    
    for task in Config.TARGET_TASKS:
        for size in Config.get_task_sizes(task):
            run_key = f"{task}_{size}"
            if run_key not in all_results:
                continue
            r = all_results[run_key]
            direct = r['direct_kd']['f1_ci']
            qf = r.get('quadratic_fit')
            
            print(f"\n  {task.upper()} (Teacher: {r['teacher_ceiling']['f1']:.4f}):")
            print(f"    Direct KD: {direct['mean']:.4f} [95%CI: {direct['lower']:.4f}-{direct['upper']:.4f}]")
            if qf:
                print(f"    Quadratic Fit: {qf['equation']}")
                print(f"    R² = {qf['r_squared']:.4f}, Optimal Depth = {qf['optimal_depth']:.2f}L")
            print(f"    {'Depth':<10} {'F1':<10} {'95% CI':<22} {'Δ vs Dir':<12} {'p':<8} {'d':<10} {'Effect':<12} {'Overhead':<10} {'Seeds'}")
            print(f"    {'─'*100}")
            
            for depth in ALL_DEPTHS:
                d = r['depth_ablation'].get(str(depth))
                if d is None:
                    continue
                ci = d['f1_ci']
                sign = "+" if d['gain_vs_direct'] >= 0 else ""
                seeds_info = f"{d.get('num_valid_seeds', '?')}/{d.get('num_total_seeds', '?')}"
                p_str = f"{d['p_value']:.4f}" if d['p_value'] is not None else "N/A"
                d_str = f"{d['cohens_d_paired']:.4f}" if d['cohens_d_paired'] is not None else "N/A"
                print(f"    {depth}L{'':<7} {ci['mean']:<10.4f} [{ci['lower']:.4f}-{ci['upper']:.4f}]   "
                      f"{sign}{d['gain_vs_direct']:<11.4f} {p_str:<8} {d_str:<10} "
                      f"{d['effect_size']:<12} {d['compute_overhead']:<10.2f}x {seeds_info}")

    print(f"\n✓ DONE")


if __name__ == "__main__":
    main()