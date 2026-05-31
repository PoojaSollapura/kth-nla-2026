# Working notes

Internal scratchpad. Not the deliverable — that's `README.md`.

## Method summary (from the paper)

Two components, both built from the target model M:

- **Verbalizer (AV):** an LLM with the same architecture as M. Takes a unit-normalized activation `h` from a middle-to-late layer of M, injected as a token embedding, and autoregressively generates a natural-language description `z`.
- **Reconstructor (AR):** first `l` layers of M with a learned affine head. Takes `z` (wrapped in a fixed prompt) and predicts `ĥ`.

Training:

- **AR:** supervised regression on MSE(`h`, AR(`z`)).
- **AV:** RL — in the paper, GRPO with reward `−log ‖h − AR(z)‖²` and a KL penalty to a warm-started init. For our small-scale version we'll use a simpler RL signal (REINFORCE-style) or iterative SFT.
- **Warm-start (AV):** paper uses Claude to generate text summaries of activation contexts → ~0.3–0.4 FVE before RL. We don't have a frontier model; we'll either (a) use a small open-source LLM to summarize, or (b) skip warm-start and document the impact.

## Metric: FVE

```
FVE = 1 − MSE(h, ĥ) / Var(h)
```

- FVE = 1 → perfect, FVE = 0 → no better than predicting mean activation.
- **Gotcha (from KTH issue #3):** if computed on unit-normalized activations, Var collapses (~2e-4) and FVE goes wildly negative even at high cosine similarity. **Decision: compute FVE on raw (unnormalized) activations.** Normalization is only used for AV injection. Report cosine similarity as a secondary metric.

## Paper headline

Anthropic reports 0.6–0.8 FVE on Claude Opus 4.6, Haiku 4.5, Haiku 3.5. We expect lower numbers on a small open-source model — and the discussion of *why* is part of what's evaluated.

## Setup decisions

- **Target model:** Pythia-160M (EleutherAI). Purpose-built for interpretability, small enough to iterate fast, has 70M / 410M / 1B siblings for a possible size-vs-FVE comparison.
- **Layer:** layer 6 of 12 (middle-to-late, matches the paper's choice in spirit).
- **Compute:** Google Colab free tier (T4 GPU).

## Open questions to resolve as we go

- Exact dataset for activation harvesting (likely a small wikitext / fineweb subset).
- Warm-start strategy (open-source LLM summarizer vs. rule-based vs. skip).
- Whether to also report a Pythia-410M run as a size-comparison data point if time allows.
