# SCI: Surgical Cognitive Interpreter

_A closed-loop metacognitive controller for signal dynamics_

This repository contains the reference implementation of **SCI**, introduced in:

> **“SCI: A Metacognitive Control for Signal Dynamics.”**  
> Author: Vishal Joshua Meesala  
> (arXiv link to be added here)

---

## 🔍 Motivation

Most neural networks behave as **open-loop function approximators**: they map inputs to outputs in a single pass, with no explicit regulation of uncertainty, clarity, or explanation quality.

**SCI** wraps existing models with a **closed-loop metacognitive controller** that:

- Monitors a scalar interpretive state $ SP(t) $
- Computes interpretive error $ \Delta SP = SP^* - SP $
- Updates interpreter parameters Θ using a Lyapunov-inspired rule
- Allocates **more inference steps to ambiguous inputs**
- Produces a **safety signal** useful for abstention, escalation, or human-in-the-loop review

---

## 🧠 Core Components

- **Decomposition (Π):** Multi-scale semantic feature bank (placeholder here)
- **Reliability weighting:** SNR / persistence / coherence-based weights (placeholder)
- **Interpreter ψΘ:** Emits markers + predictions and supports SP evaluation
- **Surgical Precision (SP):** Normalized entropy-based clarity measure
- **SCI Controller:** Closed-loop update driven by ΔSP

---

## ⚙️ Minimal Usage

```python
import torch
from sci import SCIController, Interpreter

batch_size = 16
feature_dim = 128

x = torch.randn(batch_size, feature_dim)

interpreter = Interpreter(feature_dim=feature_dim, num_markers=8, num_classes=10)
controller = SCIController(interpreter, sp_target=0.9, eta=0.01, gamma=0.1)

pred, sp, d_sp, updated = controller(x)

print("Pred logits:", pred.shape)   # (batch_size, num_classes)
print("SP:", float(sp))
print("ΔSP:", float(d_sp))
```

📦 Hugging Face Hub

This repository is intended to be integrated with the Hugging Face Hub:

Model repo: vishal-1344/sci

Planned:

PyTorchModelHubMixin support

push_to_hub.py script

Interactive SCI demo on Spaces (ZeroGPU)

📁 Layout

sci/: core library (controller, interpreter, SP, decomposition, reliability)

configs/: example configuration files (MNIST, MIT-BIH, Bearings)

examples/: Jupyter demos (to be added)

scripts/: training and Hub utilities

🔒 License

MIT License (see LICENSE).


---

**File: `scripts/push_to_hub.py`**

```python
from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    repo_id = "vishal-1344/sci"  # adjust if needed
    api = HfApi()

    # Create repo if it doesn't exist
    create_repo(repo_id, exist_ok=True, repo_type="model")

    # Upload entire project folder
    upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial SCI framework push",
    )


if __name__ == "__main__":
    main()

```

File: .gitignore

```
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv
venv/
.envrc
.ipynb_checkpoints/
dist/
build/
*.egg-info/

```
