# Thesis prose drafts — pipeline defects and observability

Draft material for Overleaf, sourced from `errors_overview.md` §6–§7. Pull into Chapter 3
(threats to validity) and Chapter 6 (limitations / further work) as needed.

---

## Chapter 3 — Threats to validity

### Silent shadow-skips and monitoring completeness

The dual-mode inference layer trades completeness for resilience: one bad shadow model
cannot kill an entire scoring cycle, because each shadow load and predict runs in an
isolated child process with a 120-second timeout. When a shadow fails transiently (for
example a DagsHub artifact download timeout), the pipeline logs a warning and continues.
No alert fires and no cycle aborts.

This design left four permanent holes in the monitoring artifacts across the tournament
(830/832 rows per cadence before reconstruction). Three further silent skips fired later
but self-healed before the affected kickoff, so they left no trace in the final
artifacts. The defect is therefore easy to miss: the pipeline appeared healthy, MLflow
runs completed, and only a row-count audit against the expected 832/832 revealed the
gaps.

For a research pipeline whose evaluation depends on a complete per-match × per-model
leaderboard, silent per-cycle holes are a direct threat to internal validity. The
mitigations — in-cycle retry on the shadow load/predict child, ERROR-level logging, and
a per-cycle model-count completeness check — are preventive only and were not shipped
because the pipeline retires at the end of the tournament.

### Push-step failures on an unattended continuous-training pipeline

An unattended pipeline that commits its own data state depends on the push step being
idempotent and retry-safe, because the remote (DagsHub for both Git pointers and DVC
blobs) is the one dependency the pipeline cannot control. Two hard crashes hit the push
gate during the tournament: a `dvc push` timeout (Jul 12 02:05) and a non-fast-forward
`git push` rejection (Jul 12 10:51). Both were transient remote failures.

The crash-and-retry pattern worked here only because the cadence was frequent (every
1–2 hours) and the data model cumulative: each retry snapshot is a superset of the
failed one, so no match rows were unrecoverable. A daily cadence or a mutable Gold
layer would have turned the same two failures into real snapshot loss. The natural
hardening — `git pull --rebase` before push and retry-with-backoff around both push
steps — was not implemented for the same retirement reason.

### Compounding observability limits

Two observability limits compound and together constitute a methodological threat rather
than a footnote.

**Killed runs are invisible to MLflow.** Every run in both outage windows reads
`FINISHED` in the tracking server — the stalled Cloud Run jobs died before reaching
`start_run`, so no record exists that they were ever attempted. Outages are detectable
only as an absence (a gap between consecutive run start times), never as a failed run.
Cloud Logging is the sole positive evidence.

**Cloud Logging expires.** Application logs before Jul 11 06:00 UTC are permanently gone
(`_Default` bucket, 30-day retention), including the Jul 10 QF matchday. Audit logs
survive 400 days in `_Required`. The R32 scheduler pause and R16 IPv6 stall therefore
sit before the retained log window and are reconstructible only because an export was
taken when it was.

An unattended pipeline whose failure mode is invisible to its experiment tracker, with a
30-day expiry on the only evidence that remains, has a real observability hole. The
recommended fix — open the MLflow run before the work begins, so a crash leaves a
`FAILED` run rather than nothing — was not shipped.

### Shadow-resolution bug (result-invalidating)

A separate defect silently invalidated frozen-mode results for three refit-eligible shadow
models (`poisson_glm`, `bayesian_poisson`, `mean_rate_poisson`): `wc_shadow` carried no
cadence alias, and frozen shadow refits wrote no `cadence_mode` tag, so
`load_shadow_model` resolved both cadences to the newest (per_round) version. The logged
"frozen" rows were byte-identical to per_round for those models. This was found only by
noticing that two columns which should differ were identical — no error, no alert, no
missing data. It leaves `xgboost` as the study's only clean frozen↔per_round contrast.
Addressed by offline frozen-shadow reconstruction (D.1) and a cadence-aware resolution
fix in `mlflow_utils.py`.

---

## Chapter 6 — Limitations and further work

### Pipeline hardening not shipped

The tournament pipeline was designed for a single competition window and retires after
the Final. Several defects were documented rather than repaired because the mitigations
pay off only on future inference cycles, of which there are none.

**Shadow load resilience.** Recommended hardening for the dual-mode inference layer:
in-cycle retry on the shadow load/predict child process, ERROR-level logging on timeout
or load failure, and a per-cycle completeness check that compares the number of models
in `predictions_all_models.csv` against the expected roster count (9 models). This would
convert silent holes into loud failures without sacrificing the champion-only fallback
that keeps the cycle alive.

**Push-step idempotency.** Recommended hardening for the DVC/Git commit gate: retry with
exponential backoff around `dvc push` and `git push`, and `git pull --rebase` before
push to avoid non-fast-forward rejections when the remote has moved. These changes are
standard practice for unattended CT pipelines but were deferred because the frequent
cadence and cumulative Gold model made the existing crash-and-retry pattern sufficient
for this tournament.

**Early MLflow run registration.** Open the MLflow run at the start of each pipeline
cycle, before ingestion and model loading, so that a task-timeout kill leaves a `FAILED`
run in the tracker rather than no record at all. This would make outage gaps visible in
MLflow without relying on log-retention windows or manual timestamp-gap analysis.

### Observability and reproducibility

The 30-day Cloud Logging retention on application logs means that operational narration
of mid-tournament incidents depends on exports taken during the tournament. Gold, DVC,
and MLflow carry everything needed for D.1 reconstruction, but the *story* of what
happened during an outage is recoverable only from preserved logs. Future deployments
should either extend log retention or treat the experiment tracker as the primary
operational audit trail.

### Model and cadence scope

After reconstruction, `xgboost` remains the only model with a clean frozen↔per_round
contrast across the full tournament. The three other refit-eligible roster shadows had
contaminated frozen columns due to the resolution bug; `mean_rate_poisson` frozen
reconstruction was dropped by decision (λ spread ~2e-3 is noise). Cross-family cadence
comparisons for RQ2 entropy therefore rely on four roster models × two modes, with the
caveat that three frozen columns required offline correction.

### Entropy trajectory gaps

Four intermediate entropy-trajectory snapshots are missing from both cadence modes (three
from the R32 scheduler pause, one from the R16 IPv6 stall). These affect RQ2 resolution
curves only; RQ1 per-match monitoring and refit gates are untouched. D.1 reconstructs
the four points by re-running inference and simulation at synthetic intermediate locked
states with the round's fixed seed.
