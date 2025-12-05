# SCI: Surgical Cognitive Interpreter  
A Metacognitive Control Layer for Signal Dynamics

This repository contains the reference implementation of the **Surgical Cognitive Interpreter (SCI)**, a closed-loop metacognitive controller that wraps existing models and turns prediction into a regulated process rather than a one-shot function evaluation.

SCI is introduced in:

**Vishal Joshua Meesala**  
*SCI: A Metacognitive Control for Signal Dynamics.*  
arXiv:2511.12240, 2025  
https://arxiv.org/abs/2511.12240

The paper formalizes interpretability as a feedback-regulated state: SCI monitors a scalar interpretive signal SP(t), defined over reliability-weighted, multi-scale features, and adaptively adjusts an interpreter’s parameters to reduce interpretive error

> ΔSP(t) = SP\*(t) − SP(t)

under Lyapunov-style stability constraints.

---

## 1. Motivation

Most neural networks are deployed as **open-loop function approximators**: they map inputs to outputs in a single forward pass, with no explicit mechanism to regulate how much computation, explanation quality, or clarification is applied to a given case. In safety–critical domains (medicine, industrial monitoring, environmental sensing), this is brittle:

- Easy and ambiguous inputs receive the same computational budget.  
- Explanations are static, post hoc, and do not adapt under drift.  
- There is no explicit notion of “interpretive error” that can be monitored and controlled.

SCI addresses this by introducing a **closed-loop metacognitive layer** that:

- Monitors a scalar interpretive state SP(t) ∈ [0, 1] over time.  
- Computes interpretive error ΔSP = SP\* − SP relative to a target clarity level SP\*.  
- Updates interpreter parameters Θ according to a Lyapunov-inspired rule with safeguards.  
- Allocates more inference steps and adaptation to ambiguous or unstable inputs.  
- Exposes ΔSP as a safety signal for abstention, escalation, or human-in-the-loop review.

Empirically, SCI:

- Allocates roughly 3.6–3.8× more computation to misclassified inputs than to correct ones.  
- Produces a scalar safety signal ΔSP with AUROC ≈ 0.70–0.86 for detecting errors across vision, medical, and industrial benchmarks.

---

## 2. Conceptual Overview

SCI is a modular architecture with the following core components.

### 2.1 Decomposition Π

A multi-scale, multimodal feature bank P(t, s) that organizes raw signals X(t) into interpretable blocks:

- Rhythmic components (frequency bands, oscillatory structure)  
- Trend components (low-frequency baselines, drifts)  
- Spatial / structural components (sensor topology, modes)  
- Cross-modal interactions (coherence, cross-correlation, causal couplings)  
- Compact but auditable latent composites Π\*

Each feature is associated with a **reliability weight** w_f(t), derived from quantities such as:

- Signal-to-noise ratio (SNR)  
- Temporal persistence  
- Multi-sensor or cross-modal coherence  

These weights allow SCI to emphasize trustworthy features and down-weight degraded sensors or spurious patterns.

### 2.2 Interpreter ψΘ

A knowledge-guided interpreter that maps the reliability-weighted feature bank into:

- **Markers** m_k: human-meaningful states or concepts  
- **Confidences** p_k(t): calibrated probabilities  
- **Rationales** r_k(t): sparse feature-level attributions and/or templated text

The interpreter can be instantiated as a modest neural head (e.g., linear layer or shallow MLP) on top of P(t, s), optionally constrained by ontologies or domain rules.

### 2.3 Surgical Precision (SP)

A scalar interpretive signal SP(t) ∈ [0, 1] that aggregates calibrated components such as:

- Clarity / selectivity  
- Pattern strength  
- Domain consistency  
- Predictive alignment

In the minimal implementation, SP is instantiated as **normalized entropy** of a marker distribution or predictive distribution: high SP corresponds to focused, confident internal usage of markers; low SP indicates diffuse or ambiguous internal state.

### 2.4 Closed-Loop Controller

A controller monitors ΔSP(t) and updates Θ accordingly. At a high level:

- Compute ΔSP(t) = SP\*(t) − SP(t) relative to a target SP\*(t).  
- If |ΔSP(t)| exceeds a threshold, update parameters:

  > Θ_{t+1} = Proj_C [ Θ_t + η_t ( ΔSP(t) · ∇_Θ SP(t) + λ_h · u_h(t) ) ]

  where:
  - η_t is a step-size schedule,  
  - λ_h is a human-gain budget,  
  - u_h(t) is a bounded human feedback signal (optional),  
  - Proj_C enforces constraints (e.g., trust region, sparsity, or parameter bounds).

- Lyapunov-style analysis shows that, under suitable conditions on η_t and λ_h, the “interpretive energy”  

  > V(t) = ½ · (ΔSP(t))²  

  decreases monotonically up to bounded noise, so explanations become more stable and consistent over time.

This yields a **reactive interpretability layer** that not only explains but also stabilizes explanations under drift, feedback, and evolving conditions.

---

## 3. Repository Structure

The repository is organized as follows:

```text
sci/                  # Core library
  __init__.py
  controller.py       # SCIController: closed-loop update over Θ using ΔSP
  interpreter.py      # Interpreter / marker head and SP computation
  sp_evaluator.py     # SP and component metrics, calibration, logging
  decomposition.py    # Decomposition Π and reliability-weighted feature bank
  reliability.py      # Reliability scores (SNR, persistence, coherence)
  utils.py            # Shared utilities and helper functions

configs/              # Example configuration files
  mnist.yaml
  mitbih.yaml
  bearings.yaml

examples/             # Jupyter notebooks (to be populated)
  mnist_sci_demo.ipynb
  ecg_sci_demo.ipynb
  bearings_sci_demo.ipynb

experiments/          # Experiment scripts, logs, and analysis

scripts/              # Training utilities, Hub utilities, etc.
  push_to_hub.py

run_sci_mitbih_fixed_k.py
run_sci_bearings.py
run_sci_signal_v2.py  # Signal-domain SCI experiments

plot_metacognition_hero.py  # Plotting script for metacognitive behavior
sc_arxiv.pdf          # Paper PDF (for convenience)
sci_latex.tex         # LaTeX source of the paper

pyproject.toml
setup.cfg
LICENSE
README.md


