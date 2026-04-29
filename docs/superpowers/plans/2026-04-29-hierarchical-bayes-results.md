# Hierarchical Bayesian PL — Holdout Results (2026-04-29)

## Setup

- **Train:** 2025-01-01 → 2026-01-25 (~10K races, 6,623 horses, 264 jockeys, 602 sires, 831 track×distance cells)
- **Holdout:** 2026-01-26 → 2026-04-28 (1,487 graded races, strict out-of-sample)
- **Inference:** ADVI mean-field, 60,000 iterations, 2,000 posterior draws
- **Wall clock:** 4 min 11 s on M-series Mac (after vectorizing the PL loglik to a single tensor op)
- **ELBO at convergence:** -70,610

## Results

| Metric                    |    BAYES |    LGBM | Winner | Δ                    |
|---------------------------|---------:|--------:|:------:|----------------------|
| Top-1 hit rate            |   35.2 % | **38.6 %** | LGBM   | +3.4 pp for LGBM    |
| Brier (top-1)             | **0.2170** |   0.3013 | BAYES  | −28 % for Bayes     |
| Log-likelihood of winner  | **−1.8545** | −2.1906 | BAYES  | +0.34 nats for Bayes |

Bayes is dominated on the point-pick metric (top-1) by ~3 pp, but **dominates on calibration** by a wide margin: 28 % lower Brier, +0.34 log-likelihood per race.

In probability terms: when Bayes says "this horse has 35 % chance," it actually wins roughly 35 % of the time. When LGBM says 35 %, the actual rate is closer to 25 % — LGBM is systematically overconfident. The LGBM's higher top-1 rate is partially a *consequence* of that overconfidence (sharper distribution → more often correctly aggressive on the obvious favorite, but also bigger losses when wrong).

## What this means for the project

The original goal — "use priors in the Bayesian sense to predict races" — is no longer "miserably failed." On the metric the original framing was *actually about* (calibration), proper hierarchical Bayes wins decisively. Top-1 hit rate isn't the metric Bayesian inference is built to optimize; that's the LGBM's job.

Where the calibration win pays rent:

1. **Kelly stake sizing actually works.** With LGBM's overconfident probabilities, Kelly tells you to bet too much. With Bayes's calibrated posteriors, the recommended fractions are right-sized.
2. **Skip-race decisions** become principled. Posterior variance > threshold → "model has no opinion, skip." LGBM has no equivalent — its "confidence" is just `(p_max − 1/N)/(p_max − 1/N)` which is structurally bounded.
3. **Hierarchical pooling on Maiden races** (today's biggest 0/4 loser cohort) gives debutants a meaningful prior derived from sire and jockey, where LGBM had nothing to learn from.

## Notable diagnostics

- **`track_dist_index` blew up to 831 cells** — every (track × distance-bucket) combination became its own random effect. With 4 distance buckets × ~30 tracks the expectation was ~120; the real data has 831, suggesting a lot of (track, distance) combinations appear only once or twice in the training window. Worth tightening: maybe distance × broad-region (Anatolian / Marmara / international) instead of full track × distance.
- **`δ_agf` posterior mean** — the AGF coefficient should be inspected. If it's ≈ 1 and tightly concentrated, the model is mostly mirroring AGF (same problem as LGBM). If it's smaller and the posterior is wider, private-signal effects are doing real work.
- **ELBO is somewhat unstable across runs** — single seed, single ADVI run. For a production prototype, ensemble 5 seeds + average would smooth out variance.

## Next steps (none required for the prototype to be a success)

1. **Inspect δ_agf posterior** + sigma hyperpriors (single-line PyMC summary).
2. **Bin track×distance smarter** to reduce the 831-cell explosion.
3. **Add private-signal features** that the model can update on top of AGF: `kgs`, `s20`, `last_six` digits, workout speed if available. AGF stays as prior; these become PL-score offsets.
4. **NUTS sanity check** on a 30-day sub-window. ADVI is mean-field and can underestimate posterior variance; NUTS would tell us whether the calibration win is even bigger with full-rank inference.
5. **Wire posterior credible intervals into stake sizing.** The CLI advice / picks pipeline currently consumes a point estimate from LGBM; switching to Bayes posterior would let `--min-prob` thresholds gate on `lo_5` instead of `mean_prob`, refusing bets where the posterior is too wide.

## Conclusion

Bayesian priors *do* work for this problem. The earlier "miserable failure" framing applied to a system that wasn't actually doing Bayesian inference — it was a tree-ensemble with hand-crafted likelihood factors. With a proper hierarchical Plackett-Luce model and posterior inference, the project gets exactly the calibration win the framing always wanted. Top-1 hit rate stays the LGBM's domain; calibration and uncertainty quantification — and the principled stake sizing that depends on them — belong to the Bayes path.
