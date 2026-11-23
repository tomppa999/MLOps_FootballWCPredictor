# 90-minute rescoring: comparison for supervision

Separate results for discussion; the existing analysis remains available. These are the same saved forecasts scored against regulation time, including stoppage time, and excluding extra time and shootouts.

## What changed

- Coverage: 104 matches. Goals change in 5; home/draw/away labels change in 5.
- Draws: 24 under the original labels; 29 after 90 minutes.
- Forecasts, probabilities, fitted models, inference IDs, cadence pairing and MD1 ties are preserved. RQ2 is not changed by this rescoring.
- Existing tests and families are reused: 10,000 resamples, seed 0; two-sided RQ1 and calibration-level tests; pointwise 95% bootstrap CIs. Holm families: 9 RPS, 9 NLL, 18 ECE/reliability tests. Exploratory trend tests remain one-sided and unadjusted.
- The original pre-binned calibration file uses different labels and is not reused. RQ3 recomputes calibration from saved per-match probabilities and the 90-minute outcomes.

Changed match scores (original → 90 minutes):

- Argentina–Cape Verde (R32, 1565179): 3–2 → 1–1; outcome changes.
- Belgium–Senegal (R32, 1567308): 3–2 → 2–2; outcome changes.
- Norway–England (QF, 1581037): 1–2 → 1–1; outcome changes.
- Argentina–Switzerland (QF, 1582681): 3–1 → 1–1; outcome changes.
- Spain–Argentina (Final, 1591866): 1–0 → 0–0; outcome changes.

## RQ1: accuracy

Differences are per-round minus frozen; negative favours refitting.

- RPS, 9 tests: raw p<0.05 3 → 2; Holm p<0.05 0 → 0.
- NLL, 9 tests: raw p<0.05 0 → 0; Holm p<0.05 0 → 0.
- Overall bayesian_poisson, NLL: original Δ -0.006320, 95% CI [-0.022506, +0.009326], raw p=0.4499, Holm p=1.0000; 90-minute Δ -0.001573, 95% CI [-0.018186, +0.014247], raw p=0.8565, Holm p=1.0000.
- Overall bayesian_poisson, RPS: original Δ -0.002602, 95% CI [-0.005500, +0.000332], raw p=0.0841, Holm p=0.5046; 90-minute Δ -0.001617, 95% CI [-0.004505, +0.001343], raw p=0.2791, Holm p=1.0000.
- Overall poisson_glm, NLL: original Δ +0.011657, 95% CI [-0.014314, +0.038118], raw p=0.3826, Holm p=1.0000; 90-minute Δ +0.016696, 95% CI [-0.009324, +0.043358], raw p=0.2116, Holm p=1.0000.
- Overall poisson_glm, RPS: original Δ -0.002907, 95% CI [-0.006540, +0.000702], raw p=0.1207, Holm p=0.5714; 90-minute Δ -0.001918, 95% CI [-0.005597, +0.001731], raw p=0.3106, Holm p=1.0000.
- Overall xgboost, NLL: original Δ -0.006322, 95% CI [-0.030327, +0.017676], raw p=0.6074, Holm p=1.0000; 90-minute Δ -0.001714, 95% CI [-0.024805, +0.021351], raw p=0.8858, Holm p=1.0000.
- Overall xgboost, RPS: original Δ -0.004355, 95% CI [-0.008300, -0.000334], raw p=0.0340, Holm p=0.2720; 90-minute Δ -0.002296, 95% CI [-0.006242, +0.001632], raw p=0.2550, Holm p=1.0000.

Raw p<0.05 under the 90-minute labels (including phase subsets):

- bayesian_poisson, group, RPS: Δ -0.004516, 95% CI [-0.007948, -0.000866], raw p=0.0172, Holm p=0.1548.
- poisson_glm, group, RPS: Δ -0.005228, 95% CI [-0.010306, -0.000276], raw p=0.0487, Holm p=0.3896.

## RQ3: calibration on the same matches

ECE and reliability use all/group/KO sets of 80/48/32 treated matches. Period views and trends remain exploratory. Negative level differences favour refitting.

- Core calibration, 18 tests: raw p<0.05 1 → 0; Holm p<0.05 1 → 0.
- bayesian_poisson, KO, ECE: 0.137994 → 0.149063; Δ +0.011069, 95% CI [-0.026365, +0.043113], raw p=0.5753, Holm p=1.0000.
- poisson_glm, KO, ECE: 0.128489 → 0.140114; Δ +0.011625, 95% CI [-0.023523, +0.038908], raw p=0.4356, Holm p=1.0000.
- xgboost, KO, ECE: 0.138356 → 0.131050; Δ -0.007306, 95% CI [-0.040905, +0.039354], raw p=0.6010, Holm p=1.0000.
- bayesian_poisson, all, ECE: 0.122859 → 0.118253; Δ -0.004605, 95% CI [-0.033414, +0.022026], raw p=0.7105, Holm p=1.0000.
- poisson_glm, all, ECE: 0.110175 → 0.122388; Δ +0.012213, 95% CI [-0.014602, +0.032700], raw p=0.2803, Holm p=1.0000.
- xgboost, all, ECE: 0.117154 → 0.116685; Δ -0.000470, 95% CI [-0.028350, +0.022779], raw p=0.9656, Holm p=1.0000.
- bayesian_poisson, group, ECE: 0.138972 → 0.124033; Δ -0.014939, 95% CI [-0.048475, +0.019769], raw p=0.4267, Holm p=1.0000.
- poisson_glm, group, ECE: 0.116351 → 0.130837; Δ +0.014487, 95% CI [-0.022776, +0.039029], raw p=0.4355, Holm p=1.0000.
- xgboost, group, ECE: 0.138162 → 0.140040; Δ +0.001877, 95% CI [-0.031781, +0.029272], raw p=0.9124, Holm p=1.0000.

Tests crossing 0.05 after Holm in either version:

- xgboost, KO, ECE: original Δ -0.026863, 95% CI [-0.044154, +0.020629], raw p=0.0001, Holm p=0.0018; 90-minute Δ -0.007306, 95% CI [-0.040905, +0.039354], raw p=0.6010, Holm p=1.0000.

## RQ3: exploratory trends

The rise test asks whether frozen calibration error increases over periods. The mitigation test asks whether the frozen slope is greater than the refitted slope. A mitigation result alone does not establish that frozen calibration deteriorated.

- bayesian_poisson, all5, ECE: frozen slope -0.000284 → -0.006899, rise p 0.7788 → 0.8986; slope gap +0.003634 → +0.001011, mitigation p 0.2401 → 0.4219.
- bayesian_poisson, all5, REL: frozen slope -0.006663 → -0.013096, rise p 0.8651 → 0.9034; slope gap +0.012626 → +0.007278, mitigation p 0.0890 → 0.2526.
- poisson_glm, all5, ECE: frozen slope +0.002785 → -0.005699, rise p 0.6965 → 0.8921; slope gap +0.008433 → +0.005482, mitigation p 0.0141 → 0.1371.
- poisson_glm, all5, REL: frozen slope -0.003972 → -0.011447, rise p 0.8510 → 0.9113; slope gap +0.016426 → +0.015031, mitigation p 0.0055 → 0.0159.
- xgboost, all5, ECE: frozen slope +0.001033 → +0.000084, rise p 0.7486 → 0.7650; slope gap +0.001005 → +0.001459, mitigation p 0.4191 → 0.3962.
- xgboost, all5, REL: frozen slope -0.000194 → +0.008232, rise p 0.7769 → 0.5969; slope gap +0.001755 → +0.006249, mitigation p 0.4158 → 0.2492.
- bayesian_poisson, treated4, ECE: frozen slope +0.016848 → +0.006032, rise p 0.2891 → 0.5844; slope gap +0.001442 → -0.001659, mitigation p 0.4434 → 0.5595.
- bayesian_poisson, treated4, REL: frozen slope +0.015333 → +0.003501, rise p 0.4664 → 0.6641; slope gap +0.017048 → +0.011245, mitigation p 0.1707 → 0.2770.
- poisson_glm, treated4, ECE: frozen slope +0.026158 → +0.012584, rise p 0.1043 → 0.4177; slope gap +0.017591 → +0.013308, mitigation p 0.0300 → 0.0901.
- poisson_glm, treated4, REL: frozen slope +0.025945 → +0.012620, rise p 0.2804 → 0.5417; slope gap +0.037154 → +0.034917, mitigation p 0.0111 → 0.0176.
- xgboost, treated4, ECE: frozen slope +0.011586 → +0.009110, rise p 0.4064 → 0.4907; slope gap -0.002520 → -0.001429, mitigation p 0.5880 → 0.5470.
- xgboost, treated4, REL: frozen slope +0.014525 → +0.024401, rise p 0.4904 → 0.3692; slope gap +0.001226 → +0.008137, mitigation p 0.4602 → 0.3106.

## Reporting scope

The 90-minute results are the current reporting version. The original results are retained only to document the scoring correction. The same dependence and exchangeability limitations and pointwise CI interpretation remain; neither the tests nor the Holm families changed.

## Reproduction and files

The current analysis is reproduced with `dvc repro -s regulation_time`. This historical comparison is optional: `dvc repro -s regulation_time_comparison`. It reads the old tables and current tables and writes `docs/tables/regulation_time_comparison/` and this note. The main analysis and figures do not require the old result tables.
