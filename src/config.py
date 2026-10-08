"""Configuration for hierarchical knowledge distillation experiments."""

import os
import torch

class Config:
    # Model configuration.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_REGISTRY = {
        "electra-base": {
            "source": "google/electra-base-discriminator",
            "directory": "electra-base",
        },
        "opt-125m": {
            "source": "facebook/opt-125m",
            "directory": "opt-125m",
        },
        "roberta-base": {
            "source": "FacebookAI/roberta-base",
            "directory": "roberta-base",
        },
        "bert-base-uncased": {
            "source": "bert-base-uncased",
            "directory": "bert-base-uncased",
        },
        "gpt2": {
            "source": "gpt2",
            "directory": "gpt2",
        },
        "gpt-neo-125M": {
            "source": "EleutherAI/gpt-neo-125M",
            "directory": "gpt-neo-125M",
        },
    }
    MODEL_NAME = os.environ.get("CENG467_MODEL", "gpt-neo-125M")
    if MODEL_NAME not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown CENG467_MODEL={MODEL_NAME!r}. "
            f"Choose one of: {', '.join(MODEL_REGISTRY)}"
        )
    MODEL_SOURCE = MODEL_REGISTRY[MODEL_NAME]["source"]
    TEACHER_MODEL = os.path.join(
        PROJECT_ROOT, "data", "models", MODEL_REGISTRY[MODEL_NAME]["directory"]
    )
    EXPECTED_TEACHER_LAYERS = 12
    STUDENT_NUM_LAYERS = 6
    STUDENT_HIDDEN_SIZE = 768
    STUDENT_NUM_HEADS = 12
    
    # Training configuration.
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2       # Effective batch size: 32.
    LEARNING_RATE = 2e-5                  # Teacher learning rate.
    STUDENT_LR = 5e-5                     # Student learning rate.
    NUM_EPOCHS = 10
    MAX_SEQ_LENGTH = 128
    
    # Domain-adaptation configuration.
    ADAPT_EPOCHS = 10
    
    # Early-stopping configuration.
    EARLY_STOPPING_PATIENCE = 10           
    EARLY_STOPPING_MIN_DELTA = 0.0         
    
    # Optimizer configuration.
    WEIGHT_DECAY = 0.01              
    WARMUP_RATIO = 0.0                     
    
    # Distillation configuration.
    TEMPERATURE = 4.0
    ALPHA = 0.5
    
    # Data configuration.
    TASKS = ["sst2"]
    TARGET_TASKS = ["sst2"]
    VAL_SPLIT = 0.2
    
    # Dataset subsampling configuration.
    DATASET_SIZES = {"sst2": 6000}  # Example subsampling size.
    
    # Filesystem paths.
    MODEL_SAVE_PATH = "./models"
    RESULTS_PATH = "./results"
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "glue")
    DATA_OFFLINE = True
    STRATIFIED = True
    
    # Computation device.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reproducibility configuration.
    SEED = 42

    @classmethod
    def get_model_path(cls, task, category, filename, seed=None, dataset_size=None):
        """Build the standard path for a saved experiment artifact."""
        base_model = os.path.basename(cls.TEACHER_MODEL)
        if dataset_size is None:
            dataset_size = str(cls.DATASET_SIZES.get(task, "full"))
        else:
            dataset_size = str(dataset_size)
        if not cls.STRATIFIED and not dataset_size.endswith("_random"):
            dataset_size = f"{dataset_size}_random"
        seed_str = str(seed) if seed is not None else str(cls.SEED)
        
        path = os.path.join(
            cls.MODEL_SAVE_PATH, 
            base_model,
            task,
            dataset_size,
            f"seed_{seed_str}",
            category
        )
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, filename)

    @classmethod
    def add_split_suffix(cls, filename):
        """Deprecated compatibility helper; artifact names remain unchanged."""
        return filename

    @classmethod
    def get_results_path(cls, filename):
        os.makedirs(cls.RESULTS_PATH, exist_ok=True)
        return os.path.join(cls.RESULTS_PATH, filename)

    @classmethod
    def is_gpt_model(cls):
        return "gpt2" in cls.TEACHER_MODEL.lower() or "gpt-neo" in cls.TEACHER_MODEL.lower()

    @classmethod
    def local_files_only(cls):
        """Keep model and tokenizer loading strictly local when offline mode is enabled."""
        return cls.DATA_OFFLINE
    
    @classmethod
    def get_task_sizes(cls, task):
        """Return the configured dataset sizes for a task."""
        if task in cls.DATASET_SIZES:
            return cls.DATASET_SIZES[task] if isinstance(cls.DATASET_SIZES[task], list) else [cls.DATASET_SIZES[task]]
        return ["full"]
    
    @classmethod
    def size_to_subsample(cls, task, size):
        """Convert a dataset size to the subsampling format used by ``prepare_data``."""
        if size == "full":
            return {}
        return {task: size if isinstance(size, int) else int(size)}