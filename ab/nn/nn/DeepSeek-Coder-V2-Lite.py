from typing import Dict, Set, Any, Optional, Iterable, Union
import torch
import torch.nn as nn
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training)


def supported_hyperparameters() -> Set[str]:
    return {
        # training
        "lr",
        "weight_decay",
        "max_grad_norm",
        "grad_accum",
        "batch_size",
        # sequence lengths
        "max_length",
        # generation
        "max_new_tokens",
        "temperature",
        "top_p",
        # quantization
        "load_in_4bit",
        # LoRA (adapter/expert)
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
    }


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:

        super().__init__()
        self.device = device

        self.model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"

        # Training hyperparams
        self.lr = float(prm.get("lr", 3e-5))
        self.weight_decay = float(prm.get("weight_decay", 0.01))
        self.max_grad_norm = float(prm.get("max_grad_norm", 1.0))
        self.grad_accum = int(prm.get("grad_accum", 8))
        self.batch_size = int(prm.get("batch_size", 1))

        # Sequence + generation
        self.max_length = int(prm.get("max_length", 2048))
        self.max_new_tokens = int(prm.get("max_new_tokens", 128))
        self.temperature = float(prm.get("temperature", 0.7))
        self.top_p = float(prm.get("top_p", 0.95))

        # Quantization
        self.load_in_4bit = bool(prm.get("load_in_4bit", True))

        # LoRA config
        self.lora_r = int(prm.get("lora_r", 16))
        self.lora_alpha = int(prm.get("lora_alpha", 32))
        self.lora_dropout = float(prm.get("lora_dropout", 0.05))
        self.lora_target_modules = prm.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0
        self.tokenizer.padding_side = "right"

        # Base model
        quant_cfg = None
        if self.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        base = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_cfg,
            torch_dtype=torch.float16,
        )

        if self.load_in_4bit:
            base = prepare_model_for_kbit_training(base)

        base.config.use_cache = False

        # Attach LoRA on top of frozen backbone
        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.lora_target_modules,
        )
        self.model = get_peft_model(base, lora_cfg)

        for name, p in self.model.named_parameters():
            p.requires_grad = ("lora_" in name)

        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        self.to(self.device)

    def forward(self, **batch) -> torch.Tensor:
        """
        Input: batch dict with input_ids, attention_mask, labels
        Return: loss (as a scalar)
        """
        outputs = self.model(**batch)
        return outputs.loss

    def train_setup(self, prm: dict) -> None:
        self.lr = float(prm.get("lr", self.lr))
        self.weight_decay = float(prm.get("weight_decay", self.weight_decay))
        self.max_grad_norm = float(prm.get("max_grad_norm", self.max_grad_norm))
        self.grad_accum = int(prm.get("grad_accum", self.grad_accum))

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters found. LoRA may not be attached correctly.")

        self.optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()

    def learn(self, train_data: Iterable[Dict[str, torch.Tensor]]) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        step = 0
        for batch in train_data:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = self.forward(**batch)
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            step += 1
