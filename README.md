# Characterizing Hierarchical Knowledge Distillation Across Model Families

This repository contains the code and result artifacts for an empirical study
of when hierarchical knowledge distillation (HKD) provides a measurable
benefit over direct knowledge distillation. The study evaluates six
approximately comparable model families across four GLUE classification tasks,
multiple training-data regimes, and five random seeds.

The central question is not only how to distill a model, but **when an
intermediate assistant model is likely to help**. The experiments compare
direct teacher-to-student distillation with a two-stage
teacher-to-assistant-to-student procedure and analyze whether predictors of HKD
benefit transfer across architectures.

## Study at a glance

| Dimension | Experimental scope |
| --- | --- |
| Model families | BERT, RoBERTa, ELECTRA, GPT-2, GPT-Neo, OPT |
| Architecture groups | Encoders and decoder-only models |
| Parameter scale | Approximately 109M--125M parameters |
| Tasks | CoLA, MRPC, RTE, SST-2 |
| Assistant depths | 1, 2, 4, 6, 8, and 10 layers |
| Seeds | 42, 52, 62, 72, and 82 |
| Training regimes | 500 to 8,551 examples, depending on the task |
| Evaluation | F1, confidence intervals, paired tests, effect sizes, and efficiency |

The paper evaluates 19 configurations per model. The exact discussion tables
and appendix calculations are available in
[`results/Result_Discussion_Tables.docx`](results/Result_Discussion_Tables.docx).

## Main findings

The analysis produces the following conclusions:

1. **No single predictor dominates across all families and data regimes.**
   Encoder and decoder architectures exhibit different relationships between
   direct distillation outcomes and HKD gains.
2. **The Utility Score transfers well to encoder models in high-resource
   settings.** A BERT-derived Utility Score reaches 88--100% prediction
   accuracy for encoder models in the high-resource regime, without
   per-architecture threshold tuning.
3. **The PE Sign rule is more effective for decoder models.** The rule
   `PE < 0` reaches 75--88% accuracy for the evaluated decoder families.
4. **Predictor reliability decreases in low-resource settings.** Core Score
   success decreases from 90% to 47%, Utility Score success from 88% to 39%,
   and PE Sign success from 79% to 52% when moving from the high-resource to
   low-resource regime.
5. **Model proximity differs by architecture group.** Encoder models form a
   relatively tight CKA cluster (0.516--0.610), whereas decoder models are
   more dispersed (0.199--0.584). The difference between final-layer and
   average-last-four-layer CKA is also larger for decoder pairs (up to 0.74)
   than for encoder pairs (up to 0.06).
6. **Assistant depth is configuration dependent.** The best depth varies by
   architecture, task, and data regime; HKD should therefore not be applied
   with a fixed assistant depth without evaluation.

These findings are empirical and descriptive. The proposed predictors are
diagnostic tools, not guarantees that HKD will improve a particular model.

## Predictors and analysis

For each configuration, let:

- `T` be the teacher F1;
- `D` be the Direct KD F1;
- `N` be the No-Distill student F1;
- `E_T` be the teacher's mean soft-label entropy.

### Core Score

```text
core = T - 2D + N
     = (T - D) + (N - D)
```

The first term is the teacher--student performance gap, while the second term
captures the degradation or improvement introduced by direct distillation. A
larger Core Score indicates more unrecovered teacher signal and a stronger
opportunity for an assistant to help. The Core Score uses architecture-specific
thresholds.

### Utility Score

```text
U = Gap × SLQ × (1 - TD) × (1 - PE)
```

with:

```text
Gap = T - D
PE  = (D - N) / (T - D)
SLQ = 1 - |E_T - 0.44| / 0.44
TD  = D / T
```

The evaluated Utility Score threshold is `U > 0.055`. The entropy target
(`0.44`) and threshold are empirical choices from the study, not universal
constants.

### PE Sign rule

For the decoder experiments, the evaluated rule is:

```text
HKD is predicted to help if PE < 0
```

This corresponds to `D < N`: direct distillation performs worse than training
the student with hard labels alone.

### Signal-to-noise analysis

The study estimates the reliability of the observed direct-distillation signal
with:

```text
SNR = |D - N| / std(D)
```

where `std(D)` is recovered from the five-seed 95% confidence interval. The
analysis treats `SNR < 1` as a regime in which the signal may be difficult to
distinguish from seed-level variability.

## Method

The implemented distillation objective transfers logits only:

```text
L = α · CE(student_logits, labels)
  + (1 − α) · T² · KL(teacher_T || student_T)
```

For Direct KD, the teacher provides the soft targets directly to the student.
For HKD, Stage 1 distills the teacher into an assistant and Stage 2 distills
the assistant into the student. Hidden-state and attention alignment are not
used by the current implementation.

The study compares three training conditions:

- **No Distill:** hard-label training only;
- **Direct KD:** direct teacher-to-student logits distillation;
- **HKD:** teacher-to-assistant-to-student logits distillation.

All evaluated configurations use five seeds. Data subsets are fixed with a
sampling seed and reused across model seeds. Deterministic cuDNN settings are
enabled where supported.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── download_assets.py
│   ├── hierarchical_knowledge_distillation_all.py
│   ├── models.py
│   ├── prepare_data.py
│   ├── proximity_analysis.py
│   ├── teacher_entropy_analysis.py
│   └── train_baseline.py
└── results/
    ├── Result_Discussion_Tables.docx
    └── proximity/
        ├── model_family_dendrogram_*.png
        └── proximity_heatmaps_*.png
```

The following local assets are intentionally excluded from version control:

- `data/`: cached GLUE datasets and downloaded base models;
- `models/`: trained teacher, assistant, and student checkpoints;
- `env/`: the local Python virtual environment.

## Installation

Python 3.10 or newer is recommended. Create and activate an isolated
environment:

```bash
python -m venv env
```

On Windows:

```powershell
.\env\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source env/bin/activate
```

Install the research dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data and model assets

The code supports the following model registry keys:

```text
electra-base
opt-125m
roberta-base
bert-base-uncased
gpt2
gpt-neo-125M
```

Select a model through `CENG467_MODEL`:

```powershell
$env:CENG467_MODEL = "bert-base-uncased"
```

Download the configured teacher model, tokenizer, and GLUE data:

```bash
python src/download_assets.py
```

The default configuration is offline-first. Set `DATA_OFFLINE = True` only
after the required assets are available locally. The local model directory is
derived from `Config.TEACHER_MODEL`, normally under `data/models/`.

## Running the experiments

### Baselines and distillation

Train the teacher and non-distilled student baselines:

```bash
python src/train_baseline.py
```

The baseline script evaluates teacher and student checkpoints and then launches
the hierarchical distillation pipeline.

Run the HKD depth ablation directly:

```bash
python src/hierarchical_knowledge_distillation_all.py
```

This evaluates Direct KD and HKD for assistant depths
`[1, 2, 4, 6, 8, 10]`, using the configured tasks, dataset sizes, and seed
list. Checkpoints are written under `models/`; serialized summaries are
written under `results/`.

### Auxiliary analyses

Teacher entropy and calibration:

```bash
python src/teacher_entropy_analysis.py
```

Representation and output-space proximity:

```bash
python src/proximity_analysis.py
```

The proximity analysis computes final-layer and average-last-four-layer CKA,
Jensen--Shannon divergence at `T = 1.0`, heatmaps, and model-family
dendrograms in `results/proximity/`.

## Configuration

The primary settings are centralized in `src/config.py`.

| Parameter | Default | Description |
| --- | ---: | --- |
| `BATCH_SIZE` | `16` | Data-loader batch size |
| `GRADIENT_ACCUMULATION_STEPS` | `2` | Effective batch size multiplier |
| `MAX_SEQ_LENGTH` | `128` | Maximum tokenized sequence length |
| `STUDENT_NUM_LAYERS` | `6` | Student depth |
| `ADAPT_EPOCHS` | `10` | Epochs per distillation stage |
| `TEMPERATURE` | `4.0` | KD temperature |
| `ALPHA` | `0.5` | Hard-label loss weight |
| `VAL_SPLIT` | `0.2` | Training-to-development split |
| `STRATIFIED` | `True` | Stratified subsampling and splitting |
| `SEED` | `42` | Default data/configuration seed |

The paper's multi-seed experiments use `[42, 52, 62, 72, 82]` in
`hierarchical_knowledge_distillation_all.py`.

For each task, 80% of the GLUE training split is used for training and 20% for
development/checkpoint selection. The GLUE validation split is used as the
final evaluation split. The full split sizes are:

| Task | Train | Development | Test |
| --- | ---: | ---: | ---: |
| CoLA | 6,841 | 1,710 | 1,043 |
| MRPC | 2,934 | 734 | 408 |
| RTE | 1,992 | 498 | 277 |
| SST-2 | 53,879 | 13,470 | 872 |

## Result artifacts

[`results/Result_Discussion_Tables.docx`](results/Result_Discussion_Tables.docx)
contains the consolidated main and appendix tables:

- all-model summary tables;
- detailed encoder and decoder comparisons;
- within-family and cross-family transfer results;
- raw experiment results;
- SNR and Direct KD variance analysis;
- rule-success summaries split by data regime.

The PNG files in [`results/proximity/`](results/proximity/) provide visual
summaries of representation and predictive similarity across model families.

## Limitations

The findings should be interpreted within the study's scope:

- The models are limited to approximately 109M--125M parameters; larger
  teacher--student capacity gaps are not evaluated.
- The task set contains four English GLUE classification tasks and does not
  establish results for generation, token-level prediction, or other domains.
- The evaluated data regimes range from 500 to 8,551 examples; behavior below
  500 examples and above 67K examples is unknown.
- GPU memory constraints (an NVIDIA RTX 3060 with 6 GB VRAM) limited the
  number of data-size and sampling combinations evaluated per architecture.
- CKA depends on representation layer, and JSD is computed at `T = 1.0`
  rather than the KD training temperature `T = 4.0`.
- Exact numerical reproducibility is not guaranteed across hardware,
  CUDA/cuDNN versions, or library versions.

## Responsible use

The predictors are intended to help researchers decide when HKD is worth
testing; they are not guarantees and should not replace empirical validation.
The proximity metrics are research diagnostics, not tools for model
attribution, plagiarism detection, or intellectual-property claims.

The experiments use English GLUE benchmarks and publicly available
pre-trained models. These datasets and web-scale pre-training corpora may
contain social and demographic biases. This repository does not measure or
mitigate fairness properties of the models.

Training and proximity analysis require substantial computation. The
experimental design aims to reduce unnecessary future distillation runs, but
the environmental cost of the reported experiments remains non-negligible.

## Citation

If you use this repository, please cite the associated paper and report the
model, task, dataset size, seed list, and configuration used in your
experiments.

## License

No license file is currently included. Add an explicit license before
redistributing the code or incorporating it into another project.
