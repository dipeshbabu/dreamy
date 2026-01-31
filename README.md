# Suppression for Language Models

**Minimizing Targeted Activations via Prompt-Side Optimization**

This repo is an extension of the original **Fluent Dreaming (EPO)** codebase, adapted for our paper/poster: **_Suppression for Language Models: Minimizing Targeted Activations via Prompt-Side Optimization_**. We introduce **EPO-MIN**, an inverted EPO objective that searches for fluent “anti-prompts” that suppress a chosen internal target without inference-time activation steering or model modifications.

---

## What’s new vs. original Fluent Dreaming (EPO)

Original EPO (“fluent dreaming”) optimizes prompts to **maximize** an internal score while keeping them fluent.

We flip the goal: we **minimize** a target internal quantity while preserving fluency using **self cross-entropy (self-XE)** as a fluency proxy.

---

## Objective (EPO-MIN)

We treat prompt search as a bi-objective problem:

- **Target suppression:** minimize a target score \( f(t) \)
- **Fluency:** minimize \( XE(t) \) (self-cross-entropy)

We use scalarization over tradeoff weights \( \lambda \):

\[
J\_\lambda(t) = -f(t) - \lambda \, XE(t)
\]

and run EPO’s population-based search with gradient-informed token proposals.

---

## Targets we support

We minimize three target families:

- **Logit margin** for a chosen token \( g \)
- **Individual MLP neuron activation** (last-token)
- **Residual-direction projection** along a contrast-derived direction (last-token residual stream)

---

## Contrast directions

We build residual directions from two contrasts:

- **Evaluation-awareness vs. neutral**
- **UPPERCASE vs. lowercase**

We compute layerwise directions \( w\_\ell \) from mean-difference / simple linear probes over last-token residual states and then use chosen layers as residual targets.

---

## Key results (from the paper/poster)

**Baseline comparison (EPO-MIN vs RANDOM vs MINSCAN):**

- **Logit margin suppression:** EPO-MIN reaches **Target = −33.0** at **XE = 6.5**, outperforming RANDOM and matching MINSCAN closely.
- **Neuron / residual targets:** Pareto fronts often collapse into near-zero basins; EPO-MIN mainly improves fluency versus baselines while still suppressing targets.

We also report two standardized operating points for compressed Pareto sets:

- **BESTTARGET@FLUENT**
- **BESTFLUENCY@NEARZERO**

---

## Limitations & intended use

- **Model scope:** Experiments in the paper focus on a tractable white-box setup (e.g., _phi-2_) and relatively short prompts. Results may change for larger models, different tokenizers, longer contexts, or different internal representations.
- **Variance & stability:** Discrete prompt search can be sensitive to random seeds and initialization. Population search and restarts help, but do not remove variance.
- **Target specificity:** Suppressing a chosen internal target (e.g., a residual direction tied to evaluation-awareness) does not guarantee downstream behavior changes are monotonic or desirable; effects can be indirect and context-dependent.
- **Safety-sensitive deployment:** Do not treat EPO-MIN as a drop-in “safety fix.” Suppressing internal features can shift model behavior in unforeseen ways. Use controlled evaluation and careful monitoring if applying beyond research experiments.
- **Data/benchmark dependence:** Baseline comparisons (RANDOM, MINSCAN) depend on the prompt pool, fluency thresholds, and evaluation protocol; changing these can affect relative performance.

Intended use: **research and analysis** of prompt-side optimization and activation suppression, including reproducing the results in our paper/poster and exploring new targets/contrasts under controlled settings.

Modules:

- `dreamy.epo`: The main EPO algorithm along with a few utilities for loading models and constructing Pareto frontiers.
- `dreamy.attribution`: Code for creating causal attribution figures.
- `dreamy.runners`: "Runners" for different optimization targets like neurons, output logits, etc.
- `dreamy.experiment`: Code we used in writing the paper for running experiments on Modal and using S3 for storage.
