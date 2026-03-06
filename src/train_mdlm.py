"""
Training script for Masked Diffusion Language Model (MDLM).

Trains a bidirectional transformer to generate text via discrete diffusion.
The forward process progressively masks tokens; the model learns to unmask them.

Usage:
    python -m train_mdlm --dataset wikitext --net dit_text_s --total_steps 50000

    # Quick test run
    python -m train_mdlm --dataset wikitext --net dit_text_t --total_steps 500 \
        --batch_size 16 --seq_length 64 --sample_every 100 --log_every 50

    # Full training with wandb logging
    python -m train_mdlm --dataset wikitext --net dit_text_s --total_steps 100000 \
        --batch_size 64 --learning_rate 3e-4 --logger wandb --use_ema
"""

import argparse
import copy
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from nanodiffusion.config.mdlm_training_config import MDLMTrainingConfig
from nanodiffusion.datasets.text_dataset import TextDataset
from nanodiffusion.models.dit_text import DiTText, DiTText_T, DiTText_S, DiTText_B
from nanodiffusion.diffusion.mdlm import MDLM
from nanodiffusion.optimizers.lr_schedule import get_cosine_schedule_with_warmup


def parse_arguments():
    parser = argparse.ArgumentParser(description="MDLM Training")

    # Dataset
    parser.add_argument("-d", "--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.05)

    # Model
    parser.add_argument("--net", type=str, default="dit_text_s",
                        choices=["dit_text_t", "dit_text_s", "dit_text_b"])
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--time_conditioning", action="store_true", default=True)
    parser.add_argument("--no_time_conditioning", dest="time_conditioning", action="store_false")

    # MDLM settings
    parser.add_argument("--sampling_eps", type=float, default=1e-3)
    parser.add_argument("--antithetic_sampling", action="store_true", default=True)

    # Sampling
    parser.add_argument("--sampling_steps", type=int, default=128)
    parser.add_argument("--sampling_strategy", type=str, default="ddpm_cache",
                        choices=["ddpm_cache", "topk"])
    parser.add_argument("--sampling_temperature", type=float, default=1.0)

    # Training
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # EMA
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_beta", type=float, default=0.9999)
    parser.add_argument("--ema_start_step", type=int, default=0)

    # Logging
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--sample_every", type=int, default=2000)
    parser.add_argument("--num_samples_for_logging", type=int, default=4)
    parser.add_argument("--validate_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none")
    parser.add_argument("--checkpoint_dir", type=str, default="logs/mdlm_train")

    return parser.parse_args()


def create_model(config: MDLMTrainingConfig, vocab_size: int, max_seq_length: int) -> DiTText:
    """Create the text DiT model based on config."""
    kwargs = dict(
        dropout=config.dropout,
        time_conditioning=config.time_conditioning,
    )

    # Allow overrides
    if config.hidden_size is not None:
        kwargs["hidden_size"] = config.hidden_size
    if config.depth is not None:
        kwargs["depth"] = config.depth
    if config.num_heads is not None:
        kwargs["num_heads"] = config.num_heads

    if config.net == "dit_text_t":
        model = DiTText_T(vocab_size, max_seq_length, **kwargs)
    elif config.net == "dit_text_s":
        model = DiTText_S(vocab_size, max_seq_length, **kwargs)
    elif config.net == "dit_text_b":
        model = DiTText_B(vocab_size, max_seq_length, **kwargs)
    else:
        raise ValueError(f"Unknown model: {config.net}")

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {config.net}, params: {num_params:.2f}M")
    return model


def load_data(config: MDLMTrainingConfig):
    """Load and split the text dataset."""
    full_dataset = TextDataset(
        dataset_name=config.dataset,
        dataset_config=config.dataset_config,
        tokenizer_name=config.tokenizer,
        seq_length=config.seq_length,
        split="train",
        max_examples=config.max_train_examples,
        cache_dir=config.cache_dir,
    )

    # Split into train/val
    train_size = int((1 - config.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset.vocab_size, full_dataset.mask_token_id, full_dataset.tokenizer


def update_ema(ema_model, model, beta):
    """Update EMA model parameters."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(beta).add_(param.data, alpha=1 - beta)


def _compute_grad_norm(model):
    """Compute total gradient norm."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


@torch.no_grad()
def generate_and_log_samples(
    mdlm: MDLM, tokenizer, config: MDLMTrainingConfig, step: int, logger: str = None
):
    """Generate sample sequences and print/log them."""
    mdlm.model.eval()
    samples = mdlm.sample(
        batch_size=config.num_samples_for_logging,
        seq_length=config.seq_length,
        num_steps=config.sampling_steps,
        strategy=config.sampling_strategy,
        temperature=config.sampling_temperature,
    )

    texts = []
    print(f"\n{'='*60}")
    print(f"Generated samples at step {step}:")
    print(f"{'='*60}")
    for i, sample in enumerate(samples):
        text = tokenizer.decode(sample.cpu().tolist(), skip_special_tokens=True)
        texts.append(text)
        # Print first 200 chars
        display = text[:200] + ("..." if len(text) > 200 else "")
        print(f"[{i}] {display}")
    print(f"{'='*60}\n")

    if logger == "wandb":
        import wandb
        table = wandb.Table(columns=["step", "sample_id", "text"])
        for i, text in enumerate(texts):
            table.add_data(step, i, text[:500])
        # Log avg length of generated text as a quality proxy
        avg_len = sum(len(t.split()) for t in texts) / max(len(texts), 1)
        wandb.log({
            "samples/generated_text": table,
            "samples/avg_word_count": avg_len,
            "step": step,
        })


@torch.no_grad()
def compute_val_loss(mdlm: MDLM, val_loader, device, max_batches: int = 20) -> float:
    """Compute average validation loss."""
    mdlm.model.eval()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x_0 = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        loss = mdlm.compute_loss(x_0, attention_mask)
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def train(config: MDLMTrainingConfig):
    """Main training function."""
    device = torch.device(config.device)

    # Set up cache directories
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))

    # Load data
    print("Loading dataset...")
    train_loader, val_loader, vocab_size, mask_token_id, tokenizer = load_data(config)
    print(f"Vocab size: {vocab_size}, Mask token ID: {mask_token_id}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = create_model(config, vocab_size, config.seq_length)
    model = model.to(device)

    if config.compile:
        model = torch.compile(model)

    # EMA model
    ema_model = copy.deepcopy(model) if config.use_ema else None

    # Create MDLM diffusion
    mdlm = MDLM(
        model=model,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        seq_length=config.seq_length,
        device=str(device),
        sampling_eps=config.sampling_eps,
        antithetic_sampling=config.antithetic_sampling,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps,
        lr_min=config.lr_min,
    )

    # Set up logging
    num_params = sum(p.numel() for p in model.parameters())
    if config.logger == "wandb":
        import wandb
        run_name = f"{config.net}_lr{config.learning_rate}_bs{config.batch_size}_seq{config.seq_length}"
        if config.use_ema:
            run_name += "_ema"
        wandb_config = vars(config).copy()
        wandb_config["num_params"] = num_params
        wandb_config["num_params_M"] = num_params / 1e6
        wandb_config["vocab_size"] = vocab_size
        wandb.init(
            project="nano-diffusion-mdlm",
            name=run_name,
            config=wandb_config,
        )

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if config.fp16 else None
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if config.fp16 else torch.amp.autocast("cuda", enabled=False)

    # Training loop
    print(f"\nStarting MDLM training for {config.total_steps} steps...")
    step = 0
    num_examples = 0
    t_start = time.time()
    tokens_since_last_log = 0

    while step < config.total_steps:
        for batch in train_loader:
            if step >= config.total_steps:
                break

            model.train()
            x_0 = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_tokens = x_0.numel()
            num_examples += x_0.shape[0]
            tokens_since_last_log += batch_tokens

            optimizer.zero_grad()

            with autocast_ctx:
                loss = mdlm.compute_loss(x_0, attention_mask)

            if scaler is not None:
                scaler.scale(loss).backward()
                if config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm).item()
                else:
                    grad_norm = _compute_grad_norm(model)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm).item()
                else:
                    grad_norm = _compute_grad_norm(model)
                optimizer.step()

            lr_scheduler.step()

            # EMA update
            if config.use_ema and ema_model is not None:
                if step >= config.ema_start_step:
                    update_ema(ema_model, model, config.ema_beta)
                else:
                    ema_model.load_state_dict(model.state_dict())

            # Logging
            if step % config.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                loss_val = loss.item()
                ppl = math.exp(min(loss_val, 20))  # cap to avoid overflow
                elapsed = time.time() - t_start
                tokens_per_sec = tokens_since_last_log / max(elapsed, 1e-6) if step > 0 else 0
                t_start = time.time()
                tokens_since_last_log = 0

                print(f"Step {step}/{config.total_steps} | Loss: {loss_val:.4f} | PPL: {ppl:.1f} | GradNorm: {grad_norm:.3f} | LR: {lr:.6f} | tok/s: {tokens_per_sec:.0f}")
                if config.logger == "wandb":
                    import wandb
                    wandb.log({
                        "train/loss": loss_val,
                        "train/perplexity": ppl,
                        "train/grad_norm": grad_norm,
                        "train/learning_rate": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/num_examples": num_examples,
                        "step": step,
                    })

            # Validation
            if step > 0 and step % config.validate_every == 0:
                val_loss = compute_val_loss(mdlm, val_loader, device)
                val_ppl = math.exp(min(val_loss, 20))
                print(f"  Val loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}")
                if config.logger == "wandb":
                    import wandb
                    wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "step": step})

            # Sample generation
            if step > 0 and step % config.sample_every == 0:
                # Use EMA model for sampling if available
                if config.use_ema and ema_model is not None:
                    mdlm.model = ema_model
                    generate_and_log_samples(mdlm, tokenizer, config, step, config.logger)
                    mdlm.model = model
                else:
                    generate_and_log_samples(mdlm, tokenizer, config, step, config.logger)

            # Save checkpoint
            if step > 0 and step % config.save_every == 0:
                ckpt_path = Path(config.checkpoint_dir) / f"model_step_{step}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")
                if config.use_ema and ema_model is not None:
                    ema_path = Path(config.checkpoint_dir) / f"ema_model_step_{step}.pth"
                    torch.save(ema_model.state_dict(), ema_path)

            step += 1

    # Final save
    if step > config.min_steps_for_final_save:
        final_path = Path(config.checkpoint_dir) / "final_model.pth"
        torch.save(model.state_dict(), final_path)
        print(f"Final model saved: {final_path}")
        if config.use_ema and ema_model is not None:
            ema_path = Path(config.checkpoint_dir) / "final_ema_model.pth"
            torch.save(ema_model.state_dict(), ema_path)

    # Final generation
    if config.use_ema and ema_model is not None:
        mdlm.model = ema_model
    generate_and_log_samples(mdlm, tokenizer, config, step, config.logger)

    if config.logger == "wandb":
        import wandb
        wandb.finish()

    print(f"\nTraining complete. Total examples: {num_examples}")
    return model


def main():
    args = parse_arguments()

    # Handle logger
    logger_val = args.logger if args.logger != "none" else None

    config = MDLMTrainingConfig(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer=args.tokenizer,
        seq_length=args.seq_length,
        max_train_examples=args.max_train_examples,
        val_split=args.val_split,
        net=args.net,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
        time_conditioning=args.time_conditioning,
        sampling_eps=args.sampling_eps,
        antithetic_sampling=args.antithetic_sampling,
        sampling_steps=args.sampling_steps,
        sampling_strategy=args.sampling_strategy,
        sampling_temperature=args.sampling_temperature,
        compile=getattr(args, "compile", False),
        fp16=args.fp16,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_min=args.lr_min,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        use_ema=args.use_ema,
        ema_beta=args.ema_beta,
        ema_start_step=args.ema_start_step,
        log_every=args.log_every,
        sample_every=args.sample_every,
        num_samples_for_logging=args.num_samples_for_logging,
        validate_every=args.validate_every,
        save_every=args.save_every,
        device=args.device,
        logger=logger_val,
        checkpoint_dir=args.checkpoint_dir,
    )

    train(config)


if __name__ == "__main__":
    main()
