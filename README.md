# SCI: Surgical Cognitive Interpreter  
A Metacognitive Control Layer for Signal Dynamics

This repository contains the reference implementation of the **Surgical Cognitive Interpreter (SCI)**,
a closed-loop metacognitive controller that wraps existing models and turns prediction into a regulated
process rather than a one-shot function evaluation.

SCI is introduced in:

**Vishal Joshua Meesala**  
*SCI: A Metacognitive Control for Signal Dynamics.*  
arXiv:2511.12240, 2025  
https://arxiv.org/abs/2511.12240

The paper formalizes interpretability as a feedback-regulated state: SCI monitors a scalar interpretive
signal `SP(t)`, defined over reliability-weighted, multi-scale features, and adaptively adjusts an
interpreter’s parameters to reduce interpretive error

`DeltaSP(t) = SP_target(t) - SP(t)`

under Lyapunov-style stability constraints.

---

## 0. What This Repository Provides

This codebase provides:

- A modular implementation of SCI’s metacognitive control loop  
- Tools to construct reliability-weighted feature decompositions for time-series and other signals  
- A controller that monitors interpretive stability and adapts parameters over time  
- Experiment scripts and utilities to reproduce key empirical findings from the paper  

SCI sits on top of existing models and signal pipelines; it does **not** replace them.  
It adds a control-theoretic layer for inference-time regulation, safety, and interpretive stability.

---

## 1. Motivation

Most neural networks are deployed as **open-loop function approximators**: they map inputs to outputs
in a single forward pass, with no explicit mechanism to regulate how much computation, explanation
quality, or clarification is applied to a given case. In safety–critical domains (medicine, industrial
monitoring, environmental sensing), this is brittle:

- Easy and ambiguous inputs receive the same computational budget  
- Explanations are static, post hoc, and do not adapt under drift  
- There is no explicit notion of “interpretive error” that can be monitored and controlled  

SCI addresses this by introducing a **closed-loop metacognitive layer** that:

- Monitors a scalar interpretive state `SP(t)` in `[0, 1]` over time  
- Computes interpretive error `DeltaSP = SP_target - SP` relative to a target clarity level `SP_target`  
- Updates interpreter parameters `Theta` according to a stability-aware rule with safeguards  
- Allocates more inference steps and adaptation to ambiguous or unstable inputs  
- Exposes `DeltaSP` as a safety signal for abstention, escalation, or human-in-the-loop review  

Empirically, SCI:

- Allocates roughly 3.6–3.8x more computation to misclassified inputs than to correct ones under matched budgets  
- Produces a scalar safety signal `DeltaSP` with AUROC around 0.70–0.86 for detecting errors across vision, medical, and industrial benchmarks  

---

## 2. Conceptual Overview

SCI is a modular architecture with four core components.

---

### 2.1 Decomposition `Pi`

A multi-scale, multimodal feature bank `P(t, s)` that organizes raw signals `X(t)` into interpretable blocks:

- Rhythmic components (frequency bands, oscillatory structure)  
- Trend components (low-frequency baselines, drifts)  
- Spatial / structural components (sensor topology, modes)  
- Cross-modal interactions (coherence, cross-correlation, causal couplings)  
- Compact but auditable latent composites `Pi_star`  

Each feature is associated with a **reliability weight** `w_f(t)`, derived from quantities such as:

- Signal-to-noise ratio (SNR)  
- Temporal persistence  
- Multi-sensor or cross-modal coherence  

These weights allow SCI to emphasize trustworthy features and down-weight degraded sensors or spurious patterns.

---

### 2.2 Interpreter `psi_Theta`

A knowledge-guided interpreter that maps the reliability-weighted feature bank into:

- **Markers** `m_k`: human-meaningful states or concepts  
- **Confidences** `p_k(t)`: calibrated probabilities  
- **Rationales** `r_k(t)`: sparse feature-level attributions and/or templated text  

The interpreter can be instantiated as a modest neural head (e.g., linear layer or shallow MLP) on top of `P(t, s)`,
optionally constrained by ontologies or domain rules.

---

### 2.3 Surgical Precision (SP)

A scalar interpretive signal `SP(t)` in `[0, 1]` that aggregates calibrated components such as:

- Clarity or selectivity  
- Pattern strength  
- Domain consistency  
- Predictive alignment  

In the minimal implementation, SP is instantiated as normalized entropy of a marker or predictive distribution:
high `SP` corresponds to focused, confident internal usage of markers; low `SP` indicates a diffuse or ambiguous
internal state.

---

### 2.4 Closed-Loop Controller

The controller monitors `DeltaSP(t)` and updates the interpreter parameters over time.

1. Compute the interpretive error:

   `DeltaSP(t) = SP_target(t) - SP(t)`

2. If `|DeltaSP(t)|` is large, update parameters:

   `Theta_{t+1} = Proj_C( Theta_t + eta_t * ( DeltaSP(t) * grad_Theta_SP(t) + lambda_h * u_h(t) ) )`

   where:

   - `eta_t` is a step-size schedule  
   - `lambda_h` is a human-gain budget  
   - `u_h(t)` is an optional, bounded human feedback signal  
   - `Proj_C` enforces constraints (e.g., trust region, sparsity, parameter bounds)  

3. Define the interpretive energy:

   `V(t) = 0.5 * ( DeltaSP(t) )^2`

Under suitable conditions on `eta_t` and `lambda_h`, the energy `V(t)` decreases monotonically up to bounded noise,
so explanations become more stable and consistent over time.

This yields a **reactive interpretability layer** that not only explains but also stabilizes explanations under drift,
feedback, and evolving conditions.

---

## 3. Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/vishal-1344/sci.git
cd sci
pip install -e .

Or install from requirements:
pip install -r requirements.txt

