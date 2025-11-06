## 🧠 ProRL — Proximal Reinforcement Optimization for Language Models

This repository implements a **Proximal Reinforcement Learning (ProRL)** training loop for causal language models such as GPT-2.
It supports **multi-sample reward optimization**, **group advantage normalization (GRPO)**, and **KL-regularized PPO-style loss**, with strong emphasis on **numerical stability** and **reproducibility**.

### ✨ Key Features

* ✅ **GRPO-style policy objective**: `min(r * A, clip(r, 1-ε_low, 1+ε_high) * A)`
* ✅ **KL penalty** to stabilize divergence from reference model
* ✅ **Multiple responses per prompt** (N-sample rollout)
* ✅ **Dynamic skip** for trivial reward groups
* ✅ **NaN-safe sampling** via custom `SafeLogitsProcessor`
* ✅ **Automatic checkpointing** every 200 steps
* ✅ **Reference model reset** for long-run stability
* ✅ **Configurable rewards** and prompt dataset
* ✅ Works on **CPU or GPU (CUDA)**

---

## 🚀 Quick Start

### 1️⃣ Clone this repository

```bash
git clone https://github.com/<your-username>/ProRL.git
cd ProRL
```

### 2️⃣ Create a Python environment

```bash
python -m venv env
source env/bin/activate      # (Linux/Mac)
env\Scripts\activate         # (Windows)
```

### 3️⃣ Install dependencies

```bash
pip install torch transformers
```

### 4️⃣ Prepare your dataset

Place a text file named `prompts.txt` in the project root, where each line is one prompt, for example:

```
Write a Python function to compute factorial.
Solve: If x + 2 = 5, what's x?
Prove that the sum of two even integers is even.
```

If no file is provided, the script uses a few built-in sample prompts.

---

## ⚙️ Configuration

All settings are defined in the `Config` dataclass at the top of `prorl_train.py`.
You can easily change parameters like:

```python
@dataclass
class Config:
    model_name: str = "gpt2"        # Hugging Face model checkpoint
    n_epochs: int = 3               # Number of training epochs
    batch_size: int = 2
    n_samples: int = 2              # Number of rollouts per prompt
    max_length: int = 128           # Max generation length
    temperature: float = 0.9        # Sampling temperature
    top_p: float = 0.9              # Nucleus sampling threshold
    kl_beta: float = 0.02           # KL penalty strength
```

You can also adjust learning rate, gradient clipping, checkpoint frequency, and more.

---

## 🧩 Running Training

```bash
python prorl_train.py
```

The script will:

* Load the model and tokenizer (`gpt2` by default)
* Sample multiple responses per prompt
* Compute rewards
* Update the model using GRPO + KL regularization
* Log training progress
* Save checkpoints in `./prorl_checkpoints`

Example log output:

```
🚀 Starting training: 5 prompts, device=cuda
Epoch 0 Step 10 | loss=0.0456 | KL=0.0032 | time=24.1s
💾 Checkpoint saved: ./prorl_checkpoints/prorl_step_200.pt
✅ Training complete. Final checkpoint saved: ./prorl_checkpoints/prorl_final.pt
```

---

## 🧮 Reward Function

The reward logic is implemented in `compute_reward()` — you can easily replace it with domain-specific evaluation, such as:

* **Math problems:** symbolic solver or unit tests
* **Code generation:** run test cases for correctness
* **Natural language tasks:** classifier/verifier model score

Example:

```python
def compute_reward(prompt, response):
    # 1 for correct code, 0 otherwise
    return 1.0 if "def" in response and "return" in response else 0.0
```

---

## ⚡ Stability Notes

This implementation includes strong numerical protections:

* **SafeLogitsProcessor** clamps invalid logits before sampling
* **NaN guards** for model parameters and gradients
* **FP32-only training** to avoid half-precision overflow
* **Deterministic CuDNN settings** for reproducibility

If you previously saw:

```
RuntimeError: probability tensor contains either inf, nan or element < 0
```

this version completely prevents that error.

---

## 🧠 Theory: GRPO / Proximal RL Objective

For each sampled response *i* with advantage *Aᵢ*,
the objective follows the **grouped PPO formulation**:

[
L = \mathbb{E}*i \Big[ \min\big(r_i A_i, \text{clip}(r_i, 1-\epsilon*{low}, 1+\epsilon_{high}) A_i \big) \Big]
- \beta_{KL} , D_{KL}(\pi_{\text{online}} || \pi_{\text{ref}})
]

where:

* ( r_i = \exp(\log p_\text{online} - \log p_\text{ref}) )
* KL penalty controls deviation from the reference model.

---

## 💾 Checkpoints

* Saved every 200 steps to `./prorl_checkpoints/`
* Includes model, reference, optimizer state, and global step
* Final checkpoint: `prorl_final.pt`

You can reload from a checkpoint as:

```python
state = torch.load("./prorl_checkpoints/prorl_final.pt")
model.load_state_dict(state["model_state"])
```

---

## 🧑‍💻 Contributing

Contributions, improvements, and bug fixes are welcome!
Feel free to:

* Submit PRs for new reward modules or datasets
* Add examples for other Hugging Face models
* Extend for multi-GPU training (`torch.nn.DataParallel` or DDP)

---

## 📚 References

* Schulman et al., *Proximal Policy Optimization Algorithms* (2017)
* Bai et al., *Constitutional AI: Harmlessness from AI Feedback* (Anthropic, 2022)
* OpenAI, *Fine-Tuning GPT Models with Reinforcement Learning from Human Feedback (RLHF)*

---

## 🪪 License

This project is released under the **MIT License**.
Feel free to use, modify, and distribute for research or educational purposes.

---

## 🌟 Acknowledgements

Developed using:

* [PyTorch](https://pytorch.org)
* [Hugging Face Transformers](https://huggingface.co/transformers)

---

Would you like me to include a **diagram** (e.g., a flowchart of the ProRL loop) in the README, rendered via Markdown/mermaid? It makes the repo look more polished and helps explain the training flow visually.
