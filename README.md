# SCI: Surgical Cognitive Interpreter  
A Metacognitive Control Layer for Signal Dynamics

This repository contains the reference implementation of the **Surgical Cognitive Interpreter (SCI)**, a closed-loop metacognitive controller that wraps existing models and turns prediction into a regulated process rather than a one-shot function evaluation. SCI is introduced in:

> **Vishal Joshua Meesala.**  
> *SCI: A Metacognitive Control for Signal Dynamics.*  
> arXiv:2511.12240, 2025.  
> https://arxiv.org/abs/2511.12240

The paper formalizes interpretability as a feedback-regulated state: SCI monitors a scalar interpretive signal \( SP(t) \), defined over reliability-weighted, multi-scale features, and adaptively adjusts an interpreter’s parameters to reduce interpretive error  
\(\Delta SP(t) = SP^\*(t) - SP(t)\) under Lyapunov-style stability constraints.:contentReference[oaicite:0]{index=0}

---

## 1. Motivation

Most neural networks are deployed as **open-loop function approximators**: they map inputs to outputs in a single forward pass, with no explicit mechanism to regulate how much computation, explanation quality, or clarification is applied to a given case. In safety–critical domains (medicine, industrial monitoring, environmental sensing), this is brittle:

- Easy and ambiguous inputs receive the same computational budget.
- Explanations are static, post hoc, and do not adapt under drift.
- There is no explicit notion of “interpretive error” that can be monitored and controlled.

SCI addresses this by introducing a **closed-loop metacognitive layer** that:

- Monitors a scalar interpretive state \( SP(t) \in [0,1] \) over time.
- Computes interpretive error \(\Delta SP = SP^\* - SP\) relative to a target clarity level \( SP^\* \).
- Updates interpreter parameters \(\Theta\) according to a Lyapunov-inspired rule with safeguards.
- Allocates more inference steps and adaptation to ambiguous or unstable inputs.
- Exposes \(\Delta SP\) as a **safety signal** for abstention, escalation, or human-in-the-loop review.

Empirically, SCI allocates \(3.6\text{–}3.8\times\) more computation to misclassified inputs than to correct ones and yields a scalar safety signal \(\Delta SP\) with AUROC \(\approx 0.70\text{–}0.86\) for detecting errors across vision, medical, and industrial benchmarks.:contentReference[oaicite:1]{index=1}

---

## 2. Conceptual Overview

SCI is a modular architecture with the following core components:

1. **Decomposition \(\Pi\)**  
   A multi-scale, multimodal feature bank \( P(t,s) \) that organizes raw signals \( X(t) \) into interpretable blocks:
   - Rhythmic components (frequency bands, oscillatory structure)  
   - Trend components (low-frequency baselines, drifts)  
   - Spatial / structural components (sensor topology, modes)  
   - Cross-modal interactions (coherence, cross-correlation, causal couplings)  
   - Compact but auditable latent composites \(\Pi^\*\)

   Each feature is associated with a **reliability weight** \( w_f(t) \) derived from SNR, persistence, and coherence scores.

2. **Interpreter \( \psi_\Theta \)**  
   A knowledge-guided interpreter that maps the reliability-weighted feature bank into:
   - Markers \( m_k \) (human-meaningful states or concepts)  
   - Confidences \( p_k(t) \)  
   - Rationales \( r_k(t) \) (sparse feature-level attributions and/or templated text)

3. **Surgical Precision \( SP \)**  
   A scalar interpretive signal \( SP(t) \in [0,1] \) that aggregates calibrated components such as:
   - Clarity / selectivity  
   - Pattern strength  
   - Domain consistency  
   - Predictive alignment

   In the minimal implementation, \( SP \) is instantiated as normalized entropy of a marker distribution or predictive distribution.

4. **Closed-Loop Controller**  
   A controller that monitors \(\Delta SP(t) = SP^\*(t) - SP(t)\) and updates \(\Theta\) via
   \[
   \Theta_{t+1} = \operatorname{Proj}_\mathcal{C}\bigl[\Theta_t + \eta_t\bigl(\Delta SP(t)\nabla_\Theta SP(t) + \lambda_h u_h(t)\bigr)\bigr],
   \]
   with:
   - Step size schedule \(\eta_t\)  
   - Human-gain budget \(\lambda_h\) and bounded feedback signal \(u_h(t)\)  
   - Trust-region and rollback safeguards  
   - Conditions ensuring monotone descent of the Lyapunov energy \(V(t) = \tfrac{1}{2}(\Delta SP(t))^2\) up to bounded noise

This yields a **reactive interpretability layer** that not only explains, but also **stabilizes** explanations under drift, feedback, and evolving conditions.

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
scripts/              # Training utilities, HF Hub utilities, etc.
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

