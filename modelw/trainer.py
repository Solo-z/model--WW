"""
Training Infrastructure for Lambda Cloud

Distributed training with DeepSpeed/FSDP support.
Optimized for multi-GPU training on Lambda Cloud instances.
"""

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from modelw.model import MIDITransformer, MIDITransformerConfig, create_model
from modelw.tokenizer import MIDITokenizer, TokenizerConfig
from modelw.dataset import LakhMIDIDataset, DatasetConfig, SessionDatasetConfig, create_dataloaders


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Model
    model_size: str = "base"  # small, base, large, xl
    
    # Data
    data_dir: str = "./data/lakh_midi"
    cache_dir: str = "./cache"
    max_seq_len: int = 2048
    
    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    
    # Optimization
    use_amp: bool = True
    grad_clip: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5000
    eval_every: int = 1000
    log_every: int = 100
    
    # Distributed
    use_ddp: bool = True
    use_deepspeed: bool = False
    
    # Logging
    wandb_project: str = "model-w"
    wandb_run_name: Optional[str] = None
    
    # Session dataset (structured JSON specs)
    session_dir: Optional[str] = None
    session_cache_dir: str = "./cache/sessions"
    session_blend_ratio: float = 0.5
    
    # Lambda Cloud specific
    num_workers: int = 8
    pin_memory: bool = True


class Trainer:
    """
    MIDI Transformer Trainer with distributed training support.
    
    Features:
    - Multi-GPU training with DDP
    - Mixed precision training
    - Gradient accumulation
    - Cosine learning rate schedule
    - Checkpointing and resume
    - W&B logging
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[MIDITransformer] = None,
        tokenizer: Optional[MIDITokenizer] = None,
    ):
        self.config = config
        
        # Setup distributed
        self.distributed = self.config.use_ddp and torch.cuda.device_count() > 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.local_rank == 0
        
        if self.distributed:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
        
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = MIDITokenizer()
        else:
            self.tokenizer = tokenizer
        
        # Initialize model
        if model is None:
            self.model = create_model(
                size=config.model_size,
                vocab_size=self.tokenizer.vocab_size,
                max_seq_len=config.max_seq_len,
                gradient_checkpointing=True,  # Save memory
            )
        else:
            self.model = model
        
        self.model = self.model.to(self.device)
        
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        
        # Get raw model for saving
        self.raw_model = self.model.module if self.distributed else self.model
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if config.use_amp else None
        
        # Metrics
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # Logging
        if self.is_main and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config),
            )
        
        self._print_info()
    
    def _print_info(self):
        """Print training information."""
        if not self.is_main:
            return
        
        num_params = self.raw_model.get_num_params()
        
        print("\n" + "="*60)
        print("MODEL-W Training")
        print("="*60)
        print(f"  Model size: {self.config.model_size}")
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        print(f"  Vocab size: {self.tokenizer.vocab_size}")
        print(f"  Max seq len: {self.config.max_seq_len}")
        print(f"  Device: {self.device}")
        print(f"  Distributed: {self.distributed} ({self.world_size} GPUs)")
        print(f"  Batch size: {self.config.batch_size} x {self.config.gradient_accumulation_steps} x {self.world_size}")
        print(f"  Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps * self.world_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Max steps: {self.config.max_steps}")
        print("="*60 + "\n")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate weight decay and no decay params
        decay_params = []
        no_decay_params = []
        
        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    
    def _create_scheduler(self):
        """Create cosine annealing scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
                return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False):
        """Save training checkpoint."""
        if not self.is_main:
            return
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if path is None:
            path = checkpoint_dir / f"checkpoint_step{self.global_step}.pt"
        else:
            path = Path(path)
        
        checkpoint = {
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": asdict(self.raw_model.config),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Save tokenizer
        self.tokenizer.save(checkpoint_dir / "tokenizer")
        
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Loaded checkpoint from {path}, step {self.global_step}")
    
    def train_step(self, batch: dict) -> float:
        """Single training step."""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Forward pass with AMP
        with autocast(enabled=self.config.use_amp):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"] / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            
            total_loss += outputs["loss"].item()
            num_batches += 1
        
        if self.distributed:
            # Aggregate across GPUs
            total_loss_tensor = torch.tensor([total_loss], device=self.device)
            num_batches_tensor = torch.tensor([num_batches], device=self.device)
            dist.all_reduce(total_loss_tensor)
            dist.all_reduce(num_batches_tensor)
            total_loss = total_loss_tensor.item()
            num_batches = num_batches_tensor.item()
        
        return total_loss / num_batches
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_path: Optional[str] = None,
    ):
        """Main training loop."""
        
        if resume_path:
            self.load_checkpoint(resume_path)
        
        accumulation_loss = 0
        start_time = time.time()
        
        # Progress bar (main process only)
        if self.is_main:
            pbar = tqdm(total=self.config.max_steps, initial=self.global_step, desc="Training")
        
        data_iter = iter(train_loader)
        
        while self.global_step < self.config.max_steps:
            
            for accum_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Reshuffle for distributed
                    if self.distributed:
                        train_loader.sampler.set_epoch(self.global_step)
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                
                loss = self.train_step(batch)
                accumulation_loss += loss
            
            # Optimizer step
            self.optimizer_step()
            self.global_step += 1
            
            # Logging
            if self.is_main and self.global_step % self.config.log_every == 0:
                avg_loss = accumulation_loss / self.config.log_every
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                samples_per_sec = self.config.batch_size * self.config.gradient_accumulation_steps * self.config.log_every / elapsed
                
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "samples/s": f"{samples_per_sec:.1f}",
                })
                
                if WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/samples_per_sec": samples_per_sec,
                        "step": self.global_step,
                    })
                
                accumulation_loss = 0
                start_time = time.time()
            
            # Evaluation
            if self.global_step % self.config.eval_every == 0:
                val_loss = self.evaluate(val_loader)
                
                if self.is_main:
                    print(f"\nStep {self.global_step}: val_loss = {val_loss:.4f}")
                    
                    if WANDB_AVAILABLE:
                        wandb.log({
                            "val/loss": val_loss,
                            "step": self.global_step,
                        })
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
            
            # Save checkpoint
            if self.is_main and self.global_step % self.config.save_every == 0:
                self.save_checkpoint()
            
            if self.is_main:
                pbar.update(1)
        
        if self.is_main:
            pbar.close()
            self.save_checkpoint()
            print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
        
        if self.distributed:
            dist.destroy_process_group()


def _merge_yaml_config(yaml_path: str, cli_overrides: dict) -> dict:
    """Load a YAML config file and merge CLI overrides on top."""
    import yaml
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    
    flat: dict = {}
    _YAML_KEY_MAP = {
        ("model", "size"): "model_size",
        ("data", "data_dir"): "data_dir",
        ("data", "max_seq_len"): "max_seq_len",
        ("training", "batch_size"): "batch_size",
        ("training", "learning_rate"): "learning_rate",
        ("training", "max_steps"): "max_steps",
        ("training", "checkpoint_dir"): "checkpoint_dir",
        ("training", "gradient_accumulation_steps"): "gradient_accumulation_steps",
        ("training", "weight_decay"): "weight_decay",
        ("training", "warmup_steps"): "warmup_steps",
        ("training", "use_amp"): "use_amp",
        ("training", "grad_clip"): "grad_clip",
        ("training", "save_every"): "save_every",
        ("training", "eval_every"): "eval_every",
        ("training", "log_every"): "log_every",
        ("logging", "wandb_project"): "wandb_project",
        ("logging", "wandb_run_name"): "wandb_run_name",
        ("data", "session_dir"): "session_dir",
        ("data", "session_blend_ratio"): "session_blend_ratio",
    }
    
    for (section, key), field_name in _YAML_KEY_MAP.items():
        if section in raw and isinstance(raw[section], dict) and key in raw[section]:
            flat[field_name] = raw[section][key]
    
    for k, v in cli_overrides.items():
        if v is not None:
            flat[k] = v
    
    return flat


def train_cli():
    """CLI for training."""
    import fire
    
    def train(
        data_dir: str = "./data/lakh_midi",
        model_size: str = "base",
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        max_steps: int = 100000,
        checkpoint_dir: str = "./checkpoints",
        resume: Optional[str] = None,
        wandb_project: str = "model-w",
        session_dir: Optional[str] = None,
        session_blend_ratio: float = 0.5,
        config_file: Optional[str] = None,
    ):
        """Train MIDI generation model.
        
        Args:
            config_file: Optional path to a YAML config. CLI flags override YAML values.
        """
        overrides = {
            "data_dir": data_dir,
            "model_size": model_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "checkpoint_dir": checkpoint_dir,
            "wandb_project": wandb_project,
            "session_dir": session_dir,
            "session_blend_ratio": session_blend_ratio,
        }
        
        if config_file is not None:
            overrides = _merge_yaml_config(config_file, overrides)
        
        config = TrainingConfig(
            data_dir=overrides["data_dir"],
            model_size=overrides["model_size"],
            batch_size=overrides["batch_size"],
            learning_rate=overrides["learning_rate"],
            max_steps=overrides["max_steps"],
            checkpoint_dir=overrides["checkpoint_dir"],
            wandb_project=overrides["wandb_project"],
            session_dir=overrides.get("session_dir"),
            session_blend_ratio=overrides.get("session_blend_ratio", 0.5),
        )
        
        trainer = Trainer(config)
        
        dataset_config = DatasetConfig(
            data_dir=config.data_dir,
            max_seq_len=config.max_seq_len,
        )
        
        session_config = None
        if config.session_dir:
            session_config = SessionDatasetConfig(
                sessions_dir=config.session_dir,
                cache_dir=config.session_cache_dir,
                max_seq_len=config.max_seq_len,
            )
        
        train_loader, val_loader = create_dataloaders(
            dataset_config,
            trainer.tokenizer,
            batch_size=batch_size,
            num_workers=config.num_workers,
            distributed=trainer.distributed,
            session_config=session_config,
            session_blend_ratio=config.session_blend_ratio,
        )
        
        trainer.train(train_loader, val_loader, resume_path=resume)
    
    fire.Fire(train)


if __name__ == "__main__":
    train_cli()

