from __future__ import annotations

import time

PROGRAM_STARTED_UNIX = time.time()
PROGRAM_STARTED_CLOCK = time.perf_counter()

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from wandb_record import experiment

from bpe import BPE
from nn_utils import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)
from transformer_module import TransformerLM


def encode_corpus_to_npy(bpe: BPE, text_path: str, npy_path: Path) -> np.ndarray:
    """Encode a text file with our BPE and cache as int32 .npy."""
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    if npy_path.is_file():
        print(f"Loading cached tokens from {npy_path}", flush=True)
        return np.load(npy_path, mmap_mode="r")

    print(f"Encoding {text_path} -> {npy_path}", flush=True)
    ids: list[int] = []
    with open(text_path, encoding="utf-8", newline="") as f:
        for i, token_id in enumerate(bpe.encode_iterable(f)):
            ids.append(token_id)
            if (i + 1) % 5_000_000 == 0:
                print(f"  encoded {i + 1} tokens...", flush=True)
    arr = np.asarray(ids, dtype=np.int32)
    np.save(npy_path, arr)
    print(f"Saved {len(arr)} tokens to {npy_path}", flush=True)
    return arr


@torch.no_grad()
def evaluate_nll(
    model: TransformerLM,
    data: np.ndarray,
    context_length: int,
    batch_size: int,
    device: str,
) -> tuple[float, int]:
    """Total NLL and token count over non-overlapping windows of the corpus."""
    model.eval()
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = 0
    # Use contiguous windows so every target token is scored once.
    # data length must be at least context_length + 1.
    total = len(data)
    starts = range(0, total - context_length, context_length)
    batch_xs: list[np.ndarray] = []
    batch_ys: list[np.ndarray] = []

    def flush():
        nonlocal nll_sum, token_count, batch_xs, batch_ys
        if not batch_xs:
            return
        x = torch.tensor(np.stack(batch_xs), dtype=torch.long, device=device)
        y = torch.tensor(np.stack(batch_ys), dtype=torch.long, device=device)
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        num = y.numel()
        nll_sum = nll_sum + loss.to(torch.float64) * num
        token_count += num
        batch_xs, batch_ys = [], []

    for start in starts:
        batch_xs.append(data[start : start + context_length])
        batch_ys.append(data[start + 1 : start + 1 + context_length])
        if len(batch_xs) >= batch_size:
            flush()
    flush()
    return float(nll_sum.item()), int(token_count)


def train_model(config, run):
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    bpe = BPE()
    bpe.load(config["bpe_prefix"])
    vocab_size = len(bpe.vocab)
    print(f"Loaded BPE vocab_size={vocab_size} from {config['bpe_prefix']}", flush=True)

    cache_dir = Path(config.get("token_cache_dir", "outputs/token_cache"))
    train_tokens = encode_corpus_to_npy(
        bpe, config["train_file"], cache_dir / "owt_train_tokens.npy"
    )
    valid_tokens = encode_corpus_to_npy(
        bpe, config["valid_file"], cache_dir / "owt_valid_tokens.npy"
    )
    # Materialize mmap arrays that will be randomly indexed during training.
    if isinstance(train_tokens, np.memmap) or getattr(train_tokens, "flags", None) is not None:
        # Keep train as memmap-friendly ndarray; np.load mmap is fine for slicing.
        pass
    print(
        f"Train tokens={len(train_tokens)}, valid tokens={len(valid_tokens)}",
        flush=True,
    )

    context_length = int(config["context_length"])
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
        device=device,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["max_lr"]),
        betas=tuple(config.get("betas", (0.9, 0.95))),
        eps=float(config.get("eps", 1e-8)),
        weight_decay=float(config.get("weight_decay", 0.1)),
    )

    ckpt_dir = Path(config.get("checkpoint_dir", "outputs/transformer_ckpts"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.pt"

    global_step = 0
    if config.get("resume_from_run") and last_ckpt.is_file():
        global_step = load_checkpoint(last_ckpt, model, optimizer)
        print(f"Resumed from {last_ckpt} at step {global_step}", flush=True)

    batch_size = int(config["batch_size"])
    max_lr = float(config["max_lr"])
    min_lr = float(config["min_lr"])
    warmup_iters = int(config["warmup_iters"])
    cosine_iters = int(config["cosine_iters"])
    grad_clip = float(config.get("grad_clip", 1.0))
    max_steps = int(config.get("max_steps", 200_000))
    # Leave headroom for final save + validation.
    reserve_seconds = float(config.get("reserve_seconds", 1200.0))
    deadline = float(config["deadline_clock"]) - reserve_seconds

    interval_nll = None
    interval_tokens = None

    model.train()
    while global_step < max_steps:
        if time.perf_counter() >= deadline:
            print(
                f"Stopping early for budget reserve at step {global_step}",
                flush=True,
            )
            break

        lr = get_lr_cosine_schedule(
            global_step, max_lr, min_lr, warmup_iters, cosine_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_tokens, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        num_tokens = y.numel()
        loss_sum = loss.detach() * num_tokens

        if interval_nll is None:
            interval_nll = torch.zeros((), device=loss_sum.device, dtype=torch.float64)
            interval_tokens = torch.zeros((), device=loss_sum.device, dtype=torch.int64)
        interval_nll.add_(loss_sum.detach().to(dtype=torch.float64))
        interval_tokens.add_(
            torch.as_tensor(num_tokens, device=loss_sum.device, dtype=torch.int64).detach()
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), grad_clip)
        optimizer.step()
        global_step += 1

        if global_step % 100 == 0:
            logged_nll = interval_nll.item()
            logged_tokens = interval_tokens.item()
            if logged_tokens > 0:
                train_ppl = math.exp(logged_nll / logged_tokens)
                elapsed = (
                    config["previous_elapsed_seconds"]
                    + time.perf_counter()
                    - PROGRAM_STARTED_CLOCK
                )
                run.log(
                    {
                        "train/ppl": train_ppl,
                        "train/step": int(global_step),
                        "train/lr": lr,
                        "time/elapsed_seconds": elapsed,
                    }
                )
                print(
                    f"step={global_step} train_ppl={train_ppl:.3f} lr={lr:.2e} "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )
            interval_nll.zero_()
            interval_tokens.zero_()

        if global_step % int(config.get("ckpt_every", 2000)) == 0:
            save_checkpoint(model, optimizer, global_step, last_ckpt)

    # Final incomplete logging interval.
    if interval_nll is not None and interval_tokens is not None and interval_tokens.item() > 0:
        logged_nll = interval_nll.item()
        logged_tokens = interval_tokens.item()
        run.log(
            {
                "train/ppl": math.exp(logged_nll / logged_tokens),
                "train/step": int(global_step),
                "time/elapsed_seconds": config["previous_elapsed_seconds"]
                + time.perf_counter()
                - PROGRAM_STARTED_CLOCK,
            }
        )

    save_checkpoint(model, optimizer, global_step, last_ckpt)
    print(f"Saved final checkpoint to {last_ckpt}", flush=True)

    val_nll_sum, val_token_count = evaluate_nll(
        model,
        valid_tokens,
        context_length=context_length,
        batch_size=batch_size,
        device=device,
    )
    print(
        f"Validation NLL sum={val_nll_sum:.4f} tokens={val_token_count} "
        f"ppl={math.exp(val_nll_sum / max(val_token_count, 1)):.4f}",
        flush=True,
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    return {
        "val_nll_sum": float(val_nll_sum),
        "val_token_count": int(val_token_count),
        "step": int(global_step),
    }


if __name__ == "__main__":
    config = {
        "train_file": "/lmfs2026/data/assignment1-data/owt_train.txt",
        "valid_file": "/lmfs2026/data/assignment1-data/owt_valid.txt",
        "bpe_prefix": "outputs/1624ca5fca15/tokenizer",
        "max_train_seconds": 21600,
        "previous_elapsed_seconds": 0.0,
        "resume_from_run": None,
        # Mild tune vs the within-6h run: keep batch=32, align cosine to ~6h steps,
        # slightly higher lr, and reserve more time for final validation.
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "context_length": 512,
        "d_model": 768,
        "num_layers": 8,
        "num_heads": 12,
        "d_ff": 2048,
        "rope_theta": 10000.0,
        "batch_size": 32,
        "max_lr": 3.5e-4,
        "min_lr": 3.5e-5,
        "warmup_iters": 2000,
        "cosine_iters": 26000,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "grad_clip": 1.0,
        "max_steps": 200000,
        "ckpt_every": 2000,
        "reserve_seconds": 1500.0,
        "token_cache_dir": "outputs/token_cache",
        "checkpoint_dir": "/ai/lmfs2026/data/assignment1/zhuziyue/transformer_ckpts",
    }
    with experiment("transformer", config) as run:
        run.define_metric("train/step")
        run.define_metric("train/ppl", step_metric="train/step")

        def utc(timestamp):
            return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()

        previous = float(config["previous_elapsed_seconds"])
        if not math.isfinite(previous) or not 0 <= previous < 21600:
            raise ValueError("此前累计耗时无效或已用完预算")
        run.summary["time/program_started_utc"] = utc(PROGRAM_STARTED_UNIX)
        run.summary["time/previous_elapsed_seconds"] = previous
        config["deadline_clock"] = PROGRAM_STARTED_CLOCK + 21600 - previous
        try:
            result = train_model(config, run)
        finally:
            ended_unix = time.time()
            current_seconds = time.perf_counter() - PROGRAM_STARTED_CLOCK
            total_seconds = previous + current_seconds
            timing = {
                "time/program_started_utc": utc(PROGRAM_STARTED_UNIX),
                "time/experiment_ended_utc": utc(ended_unix),
                "time/current_run_seconds": current_seconds,
                "time/total_seconds": total_seconds,
                "time/difference_from_6h_seconds": total_seconds - 21600,
                "time/within_6h": total_seconds <= 21600,
            }
            run.summary.update(timing)
            print(json.dumps(timing, ensure_ascii=False), flush=True)
        if result["val_token_count"] <= 0:
            raise ValueError("验证 token 数必须大于零")
        val_loss = result["val_nll_sum"] / result["val_token_count"]
        if not math.isfinite(val_loss) or val_loss < 0:
            raise ValueError("验证平均 NLL 非有限值或为负，请检查计算")
        try:
            final_ppl = math.exp(val_loss)
        except OverflowError as error:
            raise ValueError("验证 PPL 溢出，请检查模型数值") from error
        if not math.isfinite(final_ppl):
            raise ValueError("PPL 非有限值，请检查模型数值")
        final = {
            "val/ppl": final_ppl,
            "val/token_count": result["val_token_count"],
            "train/final_step": result["step"],
            "train/total_seconds": total_seconds,
        }
        run.summary.update(final)
        print(json.dumps(final, ensure_ascii=False), flush=True)
        if not 0 <= total_seconds <= 21600:
            raise ValueError("累计训练耗时超过预算或无效，请核查")
