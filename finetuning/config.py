from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class LoRAConfig:
    r: int = 256
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    )
    task_type: str = "CAUSAL_LM"


@dataclass
class CustomTrainingArguments:
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    evaluation_strategy: str = "steps"
    do_train: bool = True
    do_eval: bool = True
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    fp16: bool = True
    save_total_limit: int = 100
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    output_dir: str = (
        f"/training_output_multiple_choices/{datetime.now().strftime('%Y%m%d')}_r_128_a_256_8B"
    )
    max_steps: int = -1
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_32bit"  # "adamw_8bit" #paged_adamw_32bit
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.01
    report_to = ["tensorboard", "mlflow"]


class Config:
    def __init__(self):
        self.lora_config = LoRAConfig()
        self.max_tokens = 400
        self.custom_eval_steps = 10000
        self.testing = False
        self.tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.model = self.tokenizer_name
        self.include_test_data = False

        self.use_bigger_dataset = True

        self.use_test_as_val = False
        if self.testing == True:
            self.training_args = TestingTrainingArguments()
        else:
            self.training_args = CustomTrainingArguments()
