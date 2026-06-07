# Natural Language Autoencoder on a Small Open-Source Model

A reimplementation and small-scale evaluation of the methodology from Anthropic's [*Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*](https://transformer-circuits.pub/2026/nla/index.html), applied to **Pythia-160M**. Submitted as the KTH PhD recruitment task for Prof. Martin Monperrus (ASSERT-KTH, May–June 2026).

> **TL;DR.** I built the verbalizer–reconstructor pair on layer 6 of Pythia-160M, trained the verbalizer with supervised next-token prediction (instead of the paper's GRPO RL), and reused/retrained the reconstructor. Final pipeline FVE = **-0.16** vs. a placeholder baseline of **-0.05**. The negative numbers are an honest small-model lower bound. The interesting finding isn't the absolute value — it's that **cosine similarity stays ≈ 0.62 throughout while FVE varies by 0.22**: direction is easy, magnitude is the bottleneck on a 160M model with simplified training.

---

## 1 — What this is and why

Modern LLMs process information as long vectors of numbers ("activations") at each internal layer. These vectors are uninterpretable — we can run the model but not read what it is "thinking." The Anthropic NLA paper proposes a translation method that is **self-verifying**: train a *verbalizer* (AV) to describe each activation in English, and a *reconstructor* (AR) to rebuild the original activation back from the description alone. If AR succeeds, AV's description must have carried the real information — words are the only channel.

Their reported FVE on Claude models is **0.6–0.8**. The task here was to reimplement the approach on a *small* open-source model, on a free GPU, and report what happens.

This is a reimplementation of the methodology, not a result-matching exercise. The interesting question is what the simplifications cost, and where the bottleneck lies when the verbalizer is itself a 160M-parameter model.

---

## 2 — Setup choices (with reasoning)

| Choice | What | Why |
|---|---|---|
| **Target model** | `EleutherAI/pythia-160m` | Smallest mainstream interpretability-native model. 12 layers, hidden size 768. Has 70M / 410M / 1B siblings I could in principle scale to. |
| **Layer** | 6 (middle of 12) | Paper specifies "middle-to-late." Layer 6 leaves enough downstream capacity for the AR's encoder while sitting late enough to capture substantive computation. |
| **Source corpus** | WikiText-2 (`wikitext-2-raw-v1`) | Small (~36k snippets), classic, fast to download. The point was the pipeline, not corpus diversity. |
| **Activation set** | 5000 samples | Each sample: a random truncation of a random snippet (16–256 tokens), with the last-token residual at layer 6 captured via forward hook. Seed 0, fully reproducible. |
| **Compute** | Google Colab free tier, T4 GPU | Everything fits; total wall-clock to reproduce all four training notebooks is well under an hour. |

### The FVE normalization trap

The paper unit-normalizes activations before injecting them into the AV. Issue [#3 on the KTH repo](https://github.com/ASSERT-KTH/phd-recruitment-2026-ai4code/issues/3) flags an important pitfall: if FVE is *also* computed on unit-normalized activations, the variance denominator collapses to ~2e-4, so even a tiny MSE produces a wildly negative FVE while cosine similarity stays at 0.96. **My implementation computes FVE on raw, unnormalized activations** (denominator ≈ 999 on my dataset). Normalization is used only for AV injection. This decision is the difference between meaningful and meaningless numbers.

---

## 3 — Implementation

The full pipeline lives in five Colab notebooks. They're meant to be run in order:

- [`notebooks/01_harvest_activations.ipynb`](notebooks/01_harvest_activations.ipynb) — harvest activations
- [`notebooks/02_reconstructor_baseline.ipynb`](notebooks/02_reconstructor_baseline.ipynb) — baseline reconstructor on placeholder descriptions
- [`notebooks/03_verbalizer.ipynb`](notebooks/03_verbalizer.ipynb) — train the verbalizer
- [`notebooks/04_retrain_reconstructor_on_av.ipynb`](notebooks/04_retrain_reconstructor_on_av.ipynb) — retrain AR on AV-generated text
- [`notebooks/05_analysis.ipynb`](notebooks/05_analysis.ipynb) — analysis figures only, no training

### 3.1 Verbalizer (AV)

**Architecture.** A trainable copy of Pythia-160M plus a single learned `Linear(768, 768)` projection. Given an activation `h`, the projection produces an *embedding-shape* vector, which is fed into Pythia as if it were the first token's embedding. Pythia then generates the description autoregressively. No textual prompt is wrapped around the activation — keeping the input minimal was simpler to debug and removed the "prompt design" confound from results.

**Training objective.** Supervised next-token cross-entropy. For each `(h, snippet)` pair, the target is the first 16 tokens of the snippet that produced `h`. The verbalizer learns to maximize `P(snippet_tokens | h)`. 4 epochs, batch 8, learning rate `1e-4`, AdamW.

**Why I simplified from the paper's GRPO RL.** GRPO requires repeated sampling from AV under a reward function backed by AR, plus a KL-to-init penalty — non-trivial to implement and debug, especially without access to the paper's hyperparameters. Cross-entropy against the source snippet is a tractable proxy that still implements "text as a learned bottleneck for activations," at the cost of losing the AR-verifies-AV guarantee. **One of this project's findings is a quantitative estimate of that cost** (Section 4).

### 3.2 Reconstructor (AR)

**Architecture.** First 6 layers of Pythia-160M (frozen) plus a learned `Linear(768, 768)` head. Given a description text, AR tokenizes it, runs it through the frozen Pythia stack, captures the layer-6 residual at the last non-padding position, and applies the linear head to produce `ĥ`.

**Training objective.** Supervised MSE between `ĥ` and the true `h`. 5 epochs, batch 32, learning rate `1e-3`, AdamW. The linear head is **initialised as the identity** — at step 0, `ĥ` equals the layer-6 activation of the description text itself, which is already a sensible starting point and speeds convergence.

### 3.3 The two-stage approximation to joint training

The paper trains AV and AR jointly. I trained them in stages: AV first (against fixed source snippets), then AR (refitted on AV's outputs in notebook 04). This is a one-sided substitute for joint training — AV never adapts to AR, but AR adapts to AV. The cost of even this simplification is what Section 4 measures.

---

## 4 — Results

### 4.1 The three-way comparison

I ran three end-to-end pipelines on the same 500-sample test split (seed 0):

| Pipeline | AR's input | FVE | cosine |
|---|---|---:|---:|
| **Baseline** (notebook 02) | First-16-token slice of the original snippet | **-0.0467** | +0.6367 |
| **Pipeline, no AR retrain** (notebook 03) | AV's generated description (using the AR head from notebook 02) | **-0.2694** | +0.6128 |
| **Pipeline, with AR retrain** (notebook 04) | AV's generated description (with a fresh AR head trained on AV outputs) | **-0.1589** | +0.6211 |

Figure: [`figures/04_pipeline_comparison.png`](figures/04_pipeline_comparison.png).

**Three observations:**

1. **All numbers are slightly negative.** This is an honest small-model lower bound. With Pythia-160M as the verbalizer instead of Claude, 16-token descriptions, and supervised SFT instead of joint RL, the pipeline cannot beat the trivial mean-prediction baseline. The Anthropic paper's 0.6–0.8 FVE comes from much larger models, much longer descriptions, and the joint RL training I did not implement. Negative FVE here doesn't say the method is wrong — it says the small-model + simplified-training regime sits below where the method works.

2. **Switching from placeholder text to AV-generated text drops FVE by 0.22** (-0.05 → -0.27). This is the *distribution-mismatch cost*: AR was trained on clean source-text prefixes, but at inference it sees AV outputs that are statistically different (different word choices, occasional weirdness from a 160M generator). AR fails on this out-of-distribution input.

3. **Retraining AR on AV's outputs recovers half of that drop** (-0.27 → -0.16). The remaining gap (≈ 0.11) is what I cannot recover *without* joint training. It is a clean quantitative estimate of what the paper's elaborate RL setup buys: AV adapting to AR, not just AR adapting to AV.

### 4.2 Cosine is stable; FVE varies — the magnitude story

Across all three pipelines, mean cosine similarity stays in a tight band: **0.61 to 0.64**. FVE varies by **0.22**. The implication is sharp: the reconstructor consistently produces vectors that point roughly the right direction, but with inconsistent magnitude. Direction is easy in this regime; magnitude calibration is hard.

This is the kind of finding you can use the verbalizer to investigate further: are the failures concentrated on activations with unusual L2 norms? On unusually short or long source snippets? I look at this in Section 5.

### 4.3 Per-sample FVE distribution

The mean is one number; the distribution tells more.

Figure: [`figures/05_fve_distribution.png`](figures/05_fve_distribution.png).

`<TODO from notebook 05 — fill in once it finishes:>`

- Number of test samples with FVE > 0: `…`
- 10th-percentile FVE: `…`
- Median FVE: `…`
- 90th-percentile FVE: `…`

### 4.4 Example reconstructions

Six examples drawn from the test set — three "best" by FVE, three "worst."

`<TODO from notebook 05 — paste the top-3 and bottom-3 (snippet, AV-description, FVE) rows:>`

| | source snippet (first 80 chars) | AV description | FVE | cos |
|---|---|---|---:|---:|
| best 1 | … | … | … | … |
| best 2 | … | … | … | … |
| best 3 | … | … | … | … |
| worst 1 | … | … | … | … |
| worst 2 | … | … | … | … |
| worst 3 | … | … | … | … |

Qualitative impression I want to call out: `<TODO: 2 sentences in Pooja's voice — do the AV descriptions look on-topic? do the bad ones look obviously broken or just slightly off?>`

---

## 5 — What surprised me

### 5.1 The bimodal L2-norm distribution

When I plotted the L2 norms of the 5000 harvested activations (notebook 01), the histogram is unmistakably bimodal — two clusters around ‖h‖ ≈ 15 and ≈ 40. I did not expect this and I do not have a definitive explanation. The most likely candidates are: (i) Pythia residual norms grow with sequence length, so longer snippets produce larger-norm activations; (ii) certain content types (citations, lists, non-Latin scripts) may produce qualitatively different residual states. Figure [`figures/05_fve_vs_length_and_norm.png`](figures/05_fve_vs_length_and_norm.png) plots per-sample FVE against ‖h‖ and against snippet length. `<TODO once notebook 05 is run: 2 sentences in Pooja's voice about whether the bad reconstructions concentrate in one of the L2 modes.>`

### 5.2 The supervised verbalizer's val loss plateau

AV training in notebook 03 shows train loss falling from 6.5 to 4.1 across 4 epochs while val loss plateaued around 5.7. That gap is mild overfitting on this small dataset (4000 training pairs), but it also explains why AV's generated descriptions are noisy at test time: it has memorized training patterns rather than learning a robust mapping. With more training data — or the paper's reward-driven training — this would presumably tighten.

### 5.3 Retraining AR helps less than I hoped

I initially expected retraining AR on AV outputs to close most of the distribution-mismatch gap, perhaps within 0.05 of baseline. It closed roughly 40% (0.11 of 0.22). That residual 0.11 is, I think, the part of the joint-training benefit that cannot be one-sided. It's a small number, but I find it scientifically informative — it suggests the supervisor's choice of RL in the paper is doing real work, not just being thorough.

---

## 6 — Limitations and what I'd do differently with more time

- **No GRPO / joint RL.** This is the biggest one. The two-stage supervised approximation provides a useful lower bound but cannot match what jointly co-adapting AV and AR delivers.
- **No frontier-model warm-start.** The paper bootstraps AV with Claude-written summaries that get them to FVE ~0.3–0.4 before RL even begins. I used truncated source text as a proxy. A defensible alternative within the small-model regime would be to use a stronger open-source LLM (e.g. Llama-3-8B) as the summarizer.
- **One model size.** Pythia siblings (70M, 410M, 1B) would let me plot FVE vs. parameter count and tell a scaling story. Time did not permit.
- **One layer.** Only layer 6 was probed. Whether activations at deeper layers (e.g. 9) reconstruct better or worse is open.
- **Greedy/sampled generation, not optimized.** I sample AV outputs at temperature 1.0 with top-k 50. A small generation-time sweep on temperature and top-k might recover some FVE cheaply.

None of these are *required* for the method to work; they are honest annotations of what the project does not yet cover.

---

## 7 — Reproducibility

All notebooks run on free-tier Google Colab with a T4 GPU. Total wall-clock to reproduce from scratch is roughly 30–40 minutes (most of it spent on the verbalizer training in notebook 03).

1. Make a Google Drive folder `kth-nla-2026/` (the notebooks mount Drive and write under that path).
2. Open notebook 01 in Colab via the GitHub integration: `colab.research.google.com → File → Open notebook → GitHub → PoojaSollapura/kth-nla-2026`. Select T4 GPU. Run all. Authorize Drive when prompted. The install cell deliberately restarts the kernel once on first run (to upgrade `sympy` past a Colab-runtime incompatibility) — when Colab reconnects, click *Run all* again.
3. Run notebooks 02, 03, 04, 05 in order, each with Run all.
4. All checkpoints and figures persist on Drive between sessions.

Everything is seeded with `SEED = 0` for reproducibility. Activations and checkpoints are excluded from the repo via `.gitignore` (they are large binary blobs); they are regenerated from the notebooks.

---

*Pooja Sollapura. Repo: [github.com/PoojaSollapura/kth-nla-2026](https://github.com/PoojaSollapura/kth-nla-2026).*
