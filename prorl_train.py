"""
ProRL Training Script (Stable & NaN-Safe)
-----------------------------------------
- Implements GRPO-like reinforcement optimization for causal LMs.
- Adds SafeLogitsProcessor to fix 'probability tensor contains inf/nan' errors.
- Includes caching, KL penalty, NaN guards, and checkpointing.

Author: ChatGPT (optimized for stable training)
"""

import os
import math
import time
import random
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)
from torch.optim import AdamW


# ---------- Config ----------
@dataclass
class Config:
    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path: str = "./prompts.txt"
    n_epochs: int = 3
    batch_size: int = 2
    n_samples: int = 2
    max_length: int = 128
    temperature: float = 0.9
    top_p: float = 0.9
    eps_low: float = 0.2
    eps_high: float = 0.4
    kl_beta: float = 0.02
    lr: float = 2e-6
    weight_decay: float = 0.01
    reset_interval_steps: int = 500
    save_dir: str = "./prorl_checkpoints"
    seed: int = 42
    skip_trivial: bool = True
    min_reward_std: float = 1e-6
    log_every: int = 10


cfg = Config()

# ---------- Setup ----------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)
os.makedirs(cfg.save_dir, exist_ok=True)

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

# ---------- Dataset ----------
class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def load_prompts_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return [
            "Prove that the sum of two even integers is even.",
            "Write a python function to compute factorial.",
            "Solve: If x + 2 = 5, what's x?",
            "Given a graph coloring problem, color a triangle.",
            "Count the letters in 'hello world'.",
        ]
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


# ---------- Reward ----------
def compute_reward(prompt: str, response: str) -> float:
    if "factorial" in prompt.lower():
        return 1.0 if "def" in response and "return" in response else 0.0
    if "prove" in prompt.lower():
        return 1.0 if "therefore" in response or "hence" in response else 0.0
    if "solve" in prompt.lower() or "what's x" in prompt.lower():
        return 1.0 if any(char.isdigit() for char in response) else 0.0
    return float(min(len(response.split()), 10)) / 10.0


# ---------- Model ----------
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
model.to(dtype=torch.float32)
ref_model.to(dtype=torch.float32)
ref_model.eval()

optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------- Utilities ----------
@torch.no_grad()
def sequence_logprob(model, tokenizer, input_text: str, response_text: str, max_length: int) -> float:
    device = next(model.parameters()).device
    full_text = input_text + tokenizer.eos_token + response_text
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_ids = enc["input_ids"]
    outputs = model(input_ids, labels=input_ids)
    logits = outputs.logits
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    log_probs = F.log_softmax(logits, dim=-1)
    prefix_enc = tokenizer(input_text + tokenizer.eos_token, return_tensors="pt").to(device)
    prefix_len = prefix_enc["input_ids"].shape[1]
    seq_len = input_ids.shape[1]
    if prefix_len >= seq_len:
        return -1e9
    token_ids = input_ids[0, prefix_len:seq_len]
    token_logp = log_probs[0, prefix_len:seq_len, :].gather(1, token_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_logp.sum().item())


def compute_group_advantages(rewards: List[float]) -> List[float]:
    r = torch.tensor(rewards, dtype=torch.float32)
    mean = r.mean()
    std = r.std(unbiased=False)
    if std.item() < cfg.min_reward_std:
        std = torch.tensor(cfg.min_reward_std)
    adv = ((r - mean) / std).tolist()
    return adv


# ---------- Safe Logits Processor ----------
class SafeLogitsProcessor(LogitsProcessor):
    """Clamps logits to avoid inf/nan sampling errors."""

    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        scores = torch.clamp(scores, -1e4, 1e4)
        return scores


# ---------- Sampling ----------
def sample_responses_and_ref_logp(prompt: str, n: int, model, ref_model, tokenizer, cfg) -> List[Dict]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()

    safe_processors = LogitsProcessorList([SafeLogitsProcessor()])

    with torch.no_grad():
        try:
            generated = model.generate(
                **inputs,
                do_sample=True,
                max_length=cfg.max_length,
                temperature=max(0.7, min(cfg.temperature, 1.0)),
                top_p=min(max(cfg.top_p, 0.8), 0.95),
                num_return_sequences=n,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=safe_processors,
                return_dict_in_generate=False,
            )
        except RuntimeError as e:
            print(f"[WARN] Generation failed for '{prompt[:30]}...': {e}")
            return []

    responses = []
    for seq in generated:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        resp_text = text[len(prompt):].strip() if text.startswith(prompt) else text
        logp_ref = sequence_logprob(ref_model, tokenizer, prompt, resp_text, cfg.max_length)
        if math.isfinite(logp_ref):
            responses.append({"response": resp_text, "logp_ref": logp_ref})

    model.train()
    return responses


# ---------- Training Loop ----------
def train_prorl():
    global optimizer
    prompts = load_prompts_from_file(cfg.dataset_path)
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    global_step = 0
    last_reset_step = 0
    ref_cache = {}

    model.train()
    print(f"🚀 Starting training: {len(dataset)} prompts, device={cfg.device}")
    start_time = time.time()

    for epoch in range(cfg.n_epochs):
        for batch_idx, batch_prompts in enumerate(dataloader):
            optimizer.zero_grad()
            batch_losses, batch_kls = [], []

            for prompt in batch_prompts:
                sampled = sample_responses_and_ref_logp(prompt, cfg.n_samples, model, ref_model, tokenizer, cfg)
                if not sampled:
                    continue

                for s in sampled:
                    s["reward"] = compute_reward(prompt, s["response"])
                rewards = [s["reward"] for s in sampled]

                if cfg.skip_trivial and all(r == rewards[0] for r in rewards):
                    continue

                advantages = compute_group_advantages(rewards)
                for s, adv in zip(sampled, advantages):
                    s["adv"] = adv

                prompt_losses, prompt_kls = [], []
                for s in sampled:
                    response = s["response"]
                    key = (prompt, response)
                    if key not in ref_cache:
                        ref_cache[key] = s["logp_ref"]
                    logp_ref = ref_cache[key]

                    enc = tokenizer(prompt + tokenizer.eos_token + response,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=cfg.max_length).to(cfg.device)
                    input_ids = enc["input_ids"]
                    outputs = model(input_ids, labels=input_ids)
                    logits = outputs.logits
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                    log_probs = F.log_softmax(logits, dim=-1)

                    prefix_len = tokenizer(prompt + tokenizer.eos_token,
                                           return_tensors="pt").to(cfg.device)["input_ids"].shape[1]
                    seq_len = input_ids.shape[1]
                    if prefix_len >= seq_len:
                        continue

                    token_ids = input_ids[0, prefix_len:seq_len]
                    token_logp = log_probs[0, prefix_len:seq_len, :].gather(1, token_ids.unsqueeze(-1)).squeeze(-1)
                    logp_online = token_logp.sum()

                    logp_ref_tensor = torch.tensor(logp_ref, device=cfg.device, dtype=logp_online.dtype)
                    r = torch.exp(torch.clamp(logp_online - logp_ref_tensor, -50, 50))
                    A = torch.tensor(s["adv"], device=cfg.device, dtype=logp_online.dtype)
                    clipped_r = torch.clamp(r, 1.0 - cfg.eps_low, 1.0 + cfg.eps_high)
                    policy_obj = torch.min(r * A, clipped_r * A)
                    rl_loss = -policy_obj
                    kl_est = (logp_online - logp_ref_tensor).clamp(min=0)
                    prompt_losses.append(rl_loss)
                    prompt_kls.append(kl_est)

                if len(prompt_losses) == 0:
                    continue

                prompt_loss = torch.stack(prompt_losses).mean()
                prompt_kl = torch.stack(prompt_kls).mean()
                total_prompt_loss = prompt_loss + cfg.kl_beta * prompt_kl
                total_prompt_loss.backward()

                # Zero NaN grads
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"[WARN] NaN grad in {name}, resetting to 0.")
                        param.grad.zero_()

                batch_losses.append(total_prompt_loss.item())
                batch_kls.append(prompt_kl.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1

            # Reset reference model periodically
            if global_step - last_reset_step >= cfg.reset_interval_steps:
                print(f"[step {global_step}] 🔁 Resetting reference model.")
                ref_model.load_state_dict(model.state_dict())
                ref_model.eval()
                optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                last_reset_step = global_step

            # Logging
            if global_step % cfg.log_every == 0:
                avg_loss = sum(batch_losses) / (len(batch_losses) + 1e-9)
                avg_kl = sum(batch_kls) / (len(batch_kls) + 1e-9)
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} Step {global_step} | loss={avg_loss:.4f} | KL={avg_kl:.4f} | time={elapsed:.1f}s")

            # Checkpoint
            if global_step % 200 == 0:
                save_path = os.path.join(cfg.save_dir, f"prorl_step_{global_step}.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "ref_state": ref_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "global_step": global_step,
                }, save_path)
                print(f"💾 Checkpoint saved: {save_path}")

    final_path = os.path.join(cfg.save_dir, "prorl_final.pt")
    torch.save({
        "model_state": model.state_dict(),
        "ref_state": ref_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
    }, final_path)
    print("✅ Training complete. Final checkpoint saved:", final_path)


# ---------- Entry ----------
if __name__ == "__main__":
    train_prorl()
