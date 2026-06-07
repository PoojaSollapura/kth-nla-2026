# Natural Language Autoencoder on a Small Open-Source Model

A reimplementation and small-scale evaluation of the methodology from Anthropic's [*Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*](https://transformer-circuits.pub/2026/nla/index.html), applied to **Pythia-160M**. Submitted as the KTH PhD recruitment task for Prof. Martin Monperrus (ASSERT-KTH, May–June 2026).

> **TL;DR.** I built the verbalizer–reconstructor pair on layer 6 of Pythia-160M, trained the verbalizer with supervised next-token prediction (instead of the paper's GRPO RL), and evaluated on 500 held-out test activations. Three findings:
>
> 1. **Median FVE = +0.74**, 99.2 % of samples reconstruct with FVE > 0 — *the mechanical pipeline works on Pythia at this scale.*
> 2. **Mean FVE = -0.16**, dragged by ~4 high-norm outlier activations (‖h‖ ≈ 350 vs. typical 25) where FVE collapses to -50…-130. Cosine on these outliers stays at 0.4–0.8 — direction is right, magnitude is catastrophically wrong. This is consistent with the documented "outlier features" phenomenon in transformer residuals.
> 3. **The AV's high-FVE descriptions are not actually topical** — they are generic WikiText-style hallucinations that AR can decode into typical-magnitude activations. Without joint RL, supervised AV training optimizes for stylistic plausibility, not faithfulness. The bottleneck is not yet doing its scientific job.

---

## 1 — What this is and why

Modern LLMs process information as long vectors of numbers ("activations") at each internal layer. These vectors are uninterpretable — we can run the model but not read what it is "thinking." The Anthropic NLA paper proposes a translation method that is **self-verifying**: train a *verbalizer* (AV) to describe each activation in English, and a *reconstructor* (AR) to rebuild the original activation back from the description alone. If AR succeeds, AV's description must have carried the real information — words are the only channel between them.

The paper reports FVE in the 0.6–0.8 range on Claude models. The task here was to reimplement the approach on a small open-source model on a free GPU and report what happens. This is a methodology reimplementation, not a number-matching exercise. The interesting question is: *does the bottleneck principle still work at small scale, and where does it break?*

---

## 2 — Setup choices

| Choice | Value | Reasoning |
|---|---|---|
| Target model | `EleutherAI/pythia-160m` | Smallest mainstream interpretability-native model. 12 layers, hidden 768. Has 70M / 410M / 1B siblings for a future scaling study. |
| Layer | 6 (middle of 12) | Paper specifies "middle-to-late"; layer 6 leaves enough downstream capacity for AR's encoder while sitting late enough to capture substantive computation. |
| Source corpus | WikiText-2 (`wikitext-2-raw-v1`) | Small, classic, fast to download. Pipeline was the point, not corpus diversity. |
| Activation set | 5000 samples | Random truncations (16–256 tokens) of random snippets, last-token residual at layer 6 captured via forward hook. Seed 0. |
| Description length | 16 tokens | Short enough for fast iteration. Longer descriptions would presumably help; this is a known knob I did not tune. |
| Compute | Colab free T4 GPU | Total wall-clock to reproduce all five notebooks: well under an hour. |

### The FVE normalization trap (from KTH issue #3)

The paper unit-normalizes activations before injecting them into the AV. Issue [#3 on the KTH repo](https://github.com/ASSERT-KTH/phd-recruitment-2026-ai4code/issues/3) flags an important pitfall: if FVE is *also* computed on unit-normalized activations, the variance denominator collapses (one applicant measured ~2e-4), and FVE goes wildly negative even at cosine ≈ 0.96. **My implementation computes FVE on raw (unnormalized) activations** (denominator ≈ 999 on my dataset). Normalization is used only to scale the activation before AV injection. This single decision is the difference between meaningful and meaningless numbers.

---

## 3 — Implementation

The full pipeline is five Colab notebooks meant to be run in order:

- [`notebooks/01_harvest_activations.ipynb`](notebooks/01_harvest_activations.ipynb) — harvest 5000 activations into Google Drive
- [`notebooks/02_reconstructor_baseline.ipynb`](notebooks/02_reconstructor_baseline.ipynb) — train AR on placeholder descriptions (first-16-token text-prefix baseline)
- [`notebooks/03_verbalizer.ipynb`](notebooks/03_verbalizer.ipynb) — train AV via supervised next-token prediction; evaluate AV → AR end-to-end
- [`notebooks/04_retrain_reconstructor_on_av.ipynb`](notebooks/04_retrain_reconstructor_on_av.ipynb) — pre-generate AV outputs for the whole dataset, then refit a fresh AR head on those outputs
- [`notebooks/05_analysis.ipynb`](notebooks/05_analysis.ipynb) — per-sample FVE distribution, best/worst examples, outlier diagnostics (no training)

### 3.1 Verbalizer (AV)

A trainable copy of Pythia-160M plus one learned `Linear(768, 768)` projection. Given an activation `h`, the projection produces an embedding-shape vector that is fed into Pythia as if it were a token embedding at position 0. Pythia generates the description autoregressively. No textual prompt is wrapped around the activation — kept input minimal to remove prompt-design confounds.

Training objective: supervised next-token cross-entropy. For each `(h, snippet)` pair, the target is the first 16 tokens of the snippet that produced `h`. 4 epochs, batch 8, LR 1e-4, AdamW.

**Why I simplified from the paper's GRPO RL.** GRPO requires repeated sampling from AV under a reward function backed by AR, plus a KL-to-init penalty. It is non-trivial to implement and tune, and the paper does not publish hyperparameters at small-model scale. Cross-entropy against the source snippet is a tractable proxy that still implements the autoencoder principle (text as a bottleneck) — at the cost of losing the AR-verifies-AV guarantee. *Quantifying that cost* is one of this project's results (Section 4).

### 3.2 Reconstructor (AR)

First 6 layers of Pythia-160M (frozen) plus a learned `Linear(768, 768)` head. Given a description, AR tokenizes it, runs it through the frozen Pythia stack, captures the layer-6 residual at the last non-padding token, and applies the head to produce `ĥ`.

Training: MSE between `ĥ` and the true `h`. 5 epochs, batch 32, LR 1e-3, AdamW. The head is initialized to the identity matrix — at step 0, `ĥ` equals the description's layer-6 representation, which is already a sensible starting point and accelerates convergence.

### 3.3 Two-stage approximation to joint training

The paper trains AV and AR jointly. I trained them in stages: AV first (against fixed source snippets), then AR (refitted on AV's outputs in notebook 04). This is a one-sided substitute — AV never adapts to AR, only AR adapts to AV. The cost of even this simplification is what Section 4 measures.

---

## 4 — Results

### 4.1 Three pipelines (pooled FVE)

Same 500-sample test split (seed 0) across all three. Pooled FVE = `1 − mean_MSE / mean_variance`.

| Pipeline | AR input | FVE (pooled) | cosine (mean) |
|---|---|---:|---:|
| Baseline (notebook 02) | First-16-token slice of source snippet | **-0.05** | +0.64 |
| AV → AR (no AR retrain) | AV's generated description | **-0.27** | +0.61 |
| AV → AR_retrain (notebook 04) | AV's generated description (AR refitted on AV outputs) | **-0.16** | +0.62 |

Figure: [`figures/04_pipeline_comparison.png`](figures/04_pipeline_comparison.png).

**What this says**, taken at face value:

- Swapping placeholder text for AV-generated text drops pooled FVE by 0.22. This is the *distribution-mismatch cost*: AR was trained on clean source-text prefixes, and AV outputs are statistically different.
- Retraining AR on AV's outputs recovers about half of that drop (0.27 → 0.16). The residual 0.11 is, I argue, an estimate of what cannot be one-sided — i.e. what the paper's joint RL is doing that staged supervised training cannot.

But pooled FVE is the wrong headline for this dataset. See 4.2.

### 4.2 The per-sample distribution tells a very different story

Pooled FVE averages errors across all samples before dividing by the average variance. **It is sensitive to a small number of high-error outliers.** When I compute FVE per sample on the 500-element test set:

| Statistic | Value |
|---|---:|
| Mean FVE | -0.159 |
| **Median FVE** | **+0.739** |
| 10th-percentile FVE | +0.622 |
| 90th-percentile FVE | +0.808 |
| Mean cosine | +0.621 |
| **Fraction with FVE > 0** | **0.992** |

Figure: [`figures/05_fve_distribution.png`](figures/05_fve_distribution.png).

**496 of the 500 test samples are reconstructed with positive FVE, most of them in the +0.6 to +0.8 band.** That is approximately the range the Anthropic paper reports on Claude — on a 160M open-source model with simplified supervised training. The four catastrophic outliers (FVE between -50 and -130) are what pull the mean to -0.16.

The naive "pooled FVE = -0.16, the method failed on Pythia" reading is wrong. **The correct reading is "the method works on 99 % of activations in this regime; a small but specific subset breaks it catastrophically."**

### 4.3 The outliers cluster at very high activation norm

The per-sample scatter of FVE against ‖h‖ (figure [`figures/05_fve_vs_length_and_norm.png`](figures/05_fve_vs_length_and_norm.png), right panel) is unambiguous. Almost all samples sit in a tight cluster at ‖h‖ ≈ 20–25 with positive FVE. The catastrophic-failure samples cluster at ‖h‖ ≈ 340–360 — more than ten times the typical magnitude. There is no overlap between the two regimes on this plot.

This is consistent with the **"outlier features" phenomenon** documented in residual streams of transformer LLMs (e.g. *Dettmers et al., LLM.int8(), 2022*; *Bondarenko et al., 2023*): a tiny fraction of activations are massively larger than the rest, dominated by a few hidden dimensions on specific (often punctuation or beginning-of-text) tokens. They are known to break quantization, low-rank approximation, and other compression schemes that work on the typical population. **My result is that the NLA's natural-language bottleneck is another such scheme that handles the typical population well and fails catastrophically on outlier-norm activations.** I think this is the most interesting finding in the project.

### 4.4 What the AV descriptions actually say (and why high FVE here is *not* what it looks like)

Pulling the highest-FVE and lowest-FVE test examples is sobering. Five of each, in the format `(snippet → AV description, FVE, cos)`:

**Top reconstructions** (highest FVE on the test set):

| Source snippet (first ~80 chars) | AV description | FVE | cos |
|---|---|---:|---:|
| `The song was included on the standard version of It Won't Be Soon Before Long...` | `During the early 1990s, Braathens became increasingly located to hear that the` | +0.86 | +0.80 |
| `The 130th Engineer Brigade Headquarters and Headquarters Company (HHC), and one of the` | `In the early 1990s and 1930s, the surrender of the surrender of the` | +0.86 | +0.80 |
| `A westward moving tropical depression developed in the southwestern Caribbean Sea...` | `As the 2004 Tour, the Grand Slam Tour chart at number 10 for the` | +0.86 | +0.76 |
| `Today, Omaha is well connected to the Interstate Highway System...` | `In the 1920s, the first Independent Flying Base WAFE moved to M` | +0.86 | +0.79 |

**Bottom reconstructions** (lowest FVE):

| Source snippet | AV description | FVE | cos |
|---|---|---:|---:|
| `In May 1940 it was reported that the area's headquarters building would change...` | `the phosphate fertilizers fertilizers (fertilizers) confer the fertilizers fertil...` | +0.28 | +0.14 |
| `The first spacecraft to explore Jupiter was Pioneer 10...` | `By the song broke into number 2 million, producing mixed reviews from APW.` | **-49.0** | +0.42 |
| `The clinical manifestations of Chagas disease are due to cell death...` | `Roxas Roxa is a chemical element with the fibarasaurus` | **-128.2** | +0.76 |
| `Florida Atlantic began its expansion beyond a one campus university in 1971...` | `The corridor of Languedek Warn's history was at a period known` | **-128.7** | +0.74 |
| `Persian agents or Palmyrene traitors: the possibility of a Persian involvement exists...` | `Following the defeat of his surrender, many SAFE was later released by the` | **-132.9** | +0.79 |

Two things are obvious from this table that the numeric summaries do not show:

**(a) The AV is not topically describing the activation.** Even the best-FVE descriptions are *off-topic*. The snippet about a song becomes a description about Braathens (a Norwegian airline); the snippet about a tropical storm becomes a description about the "2004 Tour Grand Slam"; the snippet about a Chagas-disease activation produces a fabricated phrase, "Roxas Roxa is a chemical element with the fibarasaurus." The AV has not learned *what `h` is about*; it has learned to *write text in the general style and topical distribution of WikiText-2*. AR can then reconstruct a typical-magnitude activation from any such wiki-style text, and on most samples that "typical-magnitude" reconstruction happens to be close enough to the original `h` to score high FVE. **The bottleneck is not yet doing its scientific job.** That is, in my view, the most important honest qualification on the +0.74 median result.

**(b) On the catastrophic outliers, cosine stays high (0.4–0.8) while FVE collapses.** Direction is right; magnitude is off by an order of magnitude. AR has learned a calibrated "typical-magnitude" prediction and cannot produce vectors at the outlier scale. This is the direction-easy, magnitude-hard story made concrete at the per-sample level.

### 4.5 Implication

What this project actually demonstrates on Pythia-160M:

1. **The mechanical pipeline works** — activations can be round-tripped through a text bottleneck with median FVE +0.74 on 99 % of samples, using only supervised training.
2. **But the verbalizer is not yet "verbalizing" in the paper's sense.** Without joint RL pressure, it converges on generic wiki-text style; high FVE comes from style-matching, not topical fidelity. A user reading the AV description would not learn what the activation is "about."
3. **A small subset of high-norm activations breaks the pipeline catastrophically.** This is consistent with the outlier-features phenomenon documented in the LLM-quantization literature.

These three together feel like a more honest picture than any single FVE number could give.

---

## 5 — What surprised me

### 5.1 The bimodal L2-norm distribution was a real signal

When I first plotted L2 norms in notebook 01, the histogram looked bimodal and I noted it as a curiosity. The per-sample FVE breakdown in notebook 05 made it diagnostic: **the small high-norm subpopulation is exactly the subpopulation where reconstruction fails.** This connects directly to the literature on residual-stream outlier features, which I was not specifically planning to engage with.

### 5.2 Pooled FVE alone would have been misleading

If I had stopped at notebook 04's pooled FVE of -0.16 and not looked at the distribution, I would have written this up as "the simplified pipeline does not beat a placeholder baseline on Pythia." That is technically true and quantitatively defensible, but it misses the actual structure. My reaction to the per-sample plot was that the headline metric in the paper is potentially fragile under activation-outlier conditions, and that small-model NLA work should report median FVE or fraction-positive alongside mean. This is not a framing the paper foregrounds, presumably because Claude's residuals have different outlier statistics; on a 160M model it dominates.

### 5.3 Examining the qualitative outputs changed my interpretation more than the numbers did

The single biggest update came from reading the top-5 examples in Section 4.4. The FVE numbers alone made the pipeline look like it was working. The descriptions revealed that AV had learned a *style*, not a *semantics* — it produces plausible WikiText-flavored prose that is topically unrelated to the source activation, yet AR can still decode such text into typical-magnitude activations and score high FVE. This is a kind of metric overfitting at the system level: AV and AR are jointly satisfying the FVE metric by both leaning on the same statistical prior over wiki text, without the bottleneck carrying real information. I think this would be a useful caution to surface in any small-model NLA reimplementation.

### 5.4 The supervised AV's train/val gap

AV training (notebook 03) shows train loss falling 6.5 → 4.1 over 4 epochs while val loss plateaued near 5.7. Mild overfitting on 4000 training pairs. Combined with greedy/sampled generation, the test-time descriptions are noisier than the teacher-forced training signal. The paper's reward-driven training presumably tightens this; staged supervised training does not.

---

## 6 — Limitations

- **No GRPO / joint RL.** Biggest one. The two-stage supervised approximation cannot match what jointly co-adapting AV and AR delivers. Estimated cost on pooled FVE: ≈ 0.11.
- **No frontier-model warm-start.** The paper bootstraps AV with Claude-written summaries. I used truncated source text as a proxy; a stronger open LLM (e.g. Llama-3-8B) as summarizer would be a defensible alternative.
- **One model size, one layer.** Pythia siblings (70M, 410M, 1B) would enable a scaling plot; deeper layers (e.g. 9) might reconstruct differently. Neither was probed.
- **No outlier-feature mitigation.** Given how clearly failures cluster at high ‖h‖, conditioning AR on a magnitude scalar is the obvious next step.

---

## 7 — Reproducibility

All notebooks run on free-tier Google Colab with a T4 GPU. Total wall-clock from scratch is roughly 30–40 minutes, dominated by AV training in notebook 03.

1. Make a Google Drive folder named `kth-nla-2026/`. The notebooks mount Drive and read/write under that path.
2. Open notebook 01 in Colab via the GitHub integration: `colab.research.google.com → File → Open notebook → GitHub → PoojaSollapura/kth-nla-2026`. Pick T4 GPU. **Run all**. Authorize Drive when prompted. The install cell deliberately restarts the kernel once on first run (to upgrade `sympy` past a Colab GPU-runtime incompatibility) — when Colab reconnects, click *Run all* again.
3. Run notebooks 02, 03, 04, 05 in order, each with *Run all*.
4. Checkpoints, activation tensors, and figures persist on Drive between sessions.

All randomness is seeded with `SEED = 0`. Activations, checkpoints, and large binary artifacts are excluded from the repo via [`.gitignore`](.gitignore); they regenerate from the notebooks.

---

## References

1. Anthropic (2026). *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations.* [transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/index.html)
2. Biderman, S. et al. (2023). *Pythia: A Suite for Analyzing LLMs Across Training and Scaling.* [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
3. Merity, S. et al. (2016). *Pointer Sentinel Mixture Models* (WikiText-2). [arXiv:1609.07843](https://arxiv.org/abs/1609.07843)
4. Dettmers, T. et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
5. Bondarenko, Y., Nagel, M., Blankevoort, T. (2023). *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* [arXiv:2306.12929](https://arxiv.org/abs/2306.12929)
6. KTH ASSERT (2026). Issue [#3 on phd-recruitment-2026-ai4code](https://github.com/ASSERT-KTH/phd-recruitment-2026-ai4code/issues/3) — clarification on FVE normalization.

---

*Pooja Sollapura. Repo: [github.com/PoojaSollapura/kth-nla-2026](https://github.com/PoojaSollapura/kth-nla-2026).*
