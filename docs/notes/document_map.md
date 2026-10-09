# Document map

Inventory of project documents for thesis / paper writing. Status codes:

| Status | Meaning |
|--------|---------|
| `canonical` | Keep current; cite or expand in writing |
| `source-only` | Mine for prose; do not extend as a living doc |
| `stale` | Pre-thesis / superseded; archive candidate |
| `generated` | DVC or pipeline output; regenerate, do not hand-edit |
| `reference` | Ops / config reference; not thesis prose |

Thesis chapter numbers follow [docs/plans/supervision_plan.md](../plans/supervision_plan.md).
Paper section names are placeholders for the future condensed paper (not the old course paper).

---

## Root

| File | Status | Feeds | Note |
|------|--------|-------|------|
| [README.md](../../README.md) | `canonical` | journal; method (Ch 3) | Public how-to-run; reproduce path for D.1 figures |
| [AGENTS.md](../../AGENTS.md) | `reference` | — | Agent/contributor constraints; not cited in thesis |
| [dvc.yaml](../../dvc.yaml) | `canonical` | method (Ch 3); timeline (Ch 4) | Pipeline + post-WC analysis stages (`rq_datasets`, `audit`, `leaderboards`, `figures`) |

---

## Plans

| File | Status | Feeds | Note |
|------|--------|-------|------|
| [docs/plans/supervision_plan.md](../plans/supervision_plan.md) | `canonical` | thesis TOC; journal | Moodle plan; RQs, TOC, meeting cadence |
| [docs/plans/thesis.md](../plans/thesis.md) | `canonical` | method (Ch 3); results (Ch 5); journal | Canonical work plan + D.1 status; RQ2 entropy table |
| [docs/plans/golive_runbook.md](../plans/golive_runbook.md) | `reference` | journal; timeline (Ch 4); incident (§5.5) | Jun 7–11 day-by-day; ephemeral once folded into `wc_live` |

---

## Notes

| File | Status | Feeds | Note |
|------|--------|-------|------|
| [docs/notes/decisions.md](decisions.md) | `canonical` | method (Ch 3); discussion (Ch 6) | Dated design rationale |
| [docs/notes/errors_overview.md](errors_overview.md) | `canonical` | method (Ch 3); incident (§5.5); discussion (Ch 6) | Cross-cutting defects + reconstruction threats |
| [docs/notes/wc_live.md](wc_live.md) | `canonical` | results (Ch 5); incident (§5.5); journal | Per-matchday live log + post-tournament D.1 |
| [docs/notes/results_pre_wc.md](results_pre_wc.md) | `canonical` | results §5.1; journal | Pre-WC model selection numbers |
| [docs/notes/writing_hooks.md](writing_hooks.md) | `canonical` | intro; discussion (Ch 6) | Framings that do not live in code |
| [docs/notes/thesis_prose.md](thesis_prose.md) | `canonical` | method (Ch 3); discussion (Ch 6) | Draft paragraphs on defects / observability |
| [docs/notes/document_map.md](document_map.md) | `canonical` | journal | This file |
| [docs/notes/chronology.md](chronology.md) | `canonical` | journal; incident (§5.5); timeline (Ch 4) | Single dated event table |

---

## Architecture / data / modeling specs

| File | Status | Feeds | Note |
|------|--------|-------|------|
| [docs/architecture.md](../architecture.md) | `canonical` | system (Ch 4); paper system | High-level pipeline architecture |
| [docs/feature_spec.md](../feature_spec.md) | `canonical` | method (Ch 3); appendix | Feature definitions; host override |
| [docs/data_sources.md](../data_sources.md) | `canonical` | method (Ch 3) | Bronze sources |
| [docs/modeling_plan.md](../modeling_plan.md) | `source-only` | method (Ch 3); system (Ch 4) | Pre-thesis candidate plan; partly superseded by live roster |
| [docs/testing_strategy.md](../testing_strategy.md) | `source-only` | method (Ch 3) | Test approach; thin |
| [docs/threats_to_validity.md](../threats_to_validity.md) | `canonical` | method (Ch 3); discussion (Ch 6) | Drift monitoring absence etc. |

---

## Literature

| File | Status | Feeds | Note |
|------|--------|-------|------|
| [docs/literature/literature_review.md](../literature/literature_review.md) | `canonical` | background (Ch 2); paper related work | Football goal-modeling survey; **no cadence/drift/CT papers** |
| [docs/literature/prediction_framing.md](../literature/prediction_framing.md) | `canonical` | method (Ch 3); background (Ch 2) | Goal-based vs result-based framing |
| [docs/literature/maher_1982.md](../literature/maher_1982.md) | `canonical` | background (Ch 2) | Notes |
| [docs/literature/ley_et_al_2019.md](../literature/ley_et_al_2019.md) | `canonical` | background (Ch 2) | Notes |
| [docs/literature/groll_et_al_2019.md](../literature/groll_et_al_2019.md) | `canonical` | background (Ch 2); results ceiling | Notes |
| [docs/literature/10.1515_jqas-2018-0060.pdf](../literature/10.1515_jqas-2018-0060.pdf) | `reference` | background (Ch 2) | Local PDF (gitignored from tracking) |
| [docs/literature/10.1515_jqas-2018-0060.docx](../literature/10.1515_jqas-2018-0060.docx) | `reference` | — | Local docx |
| [docs/literature/8637571.pdf](../literature/8637571.pdf) | `reference` | background (Ch 2) | Local PDF |
| [docs/literature/MLOps_Pipeline_Development_The_OSSARA_Use_Case.pdf](../literature/MLOps_Pipeline_Development_The_OSSARA_Use_Case.pdf) | `reference` | background §2.4 | Local PDF |

---

## Report / paper (LaTeX)

| File | Status | Feeds | Note |
|------|--------|-------|------|
| [docs/report/paper.tex](../report/paper.tex) | `stale` | — | Pre-cadence course paper; title and "nine models" outdated — **do not edit into new paper** |
| [docs/report/wc_mlops_report.tex](../report/wc_mlops_report.tex) | `source-only` | system (Ch 4); timeline (Ch 4) | Long course report; mine for prose + Gantt; archive candidate |
| [docs/report/references.bib](../report/references.bib) | `canonical` | all writing | 16 entries; football + general MLOps only — expand for paper |

---

## Figures

| File | Status | Feeds | Note |
|------|--------|-------|------|
| [docs/figures/entropy-trajectory.png](../figures/entropy-trajectory.png) | `generated` | results RQ2; paper | DVC `figures` stage |
| [docs/figures/entropy-stages.png](../figures/entropy-stages.png) | `generated` | results RQ2; paper | DVC `figures` stage; 2×3 stage panels |
| [docs/figures/calibration-group-vs-ko.png](../figures/calibration-group-vs-ko.png) | `generated` | results RQ3; paper | DVC `figures` stage |
| [docs/figures/Architecture.png](../figures/Architecture.png) | `source-only` | system (Ch 4) | Pre-thesis diagram |
| [docs/figures/Architecture_final.png](../figures/Architecture_final.png) | `source-only` | system (Ch 4); paper | Prefer over Architecture.png |
| [docs/figures/Process.png](../figures/Process.png) | `source-only` | system (Ch 4) | Pre-thesis |
| [docs/figures/architecture_ct_pipeline.mmd](../figures/architecture_ct_pipeline.mmd) | `canonical` | system (Ch 4) | Mermaid source for CT pipeline |
| [docs/figures/architecture_ct_pipeline.png](../figures/architecture_ct_pipeline.png) | `source-only` | system (Ch 4) | Render of `.mmd` |
| [docs/figures/cluster_experiment_v2.png](../figures/cluster_experiment_v2.png) | `stale` | — | Clustering dropped from features |
| [docs/figures/cluster_k_selection.png](../figures/cluster_k_selection.png) | `stale` | — | Clustering dropped |
| [docs/figures/feature_importance_top3.png](../figures/feature_importance_top3.png) | `source-only` | results §5.1 | Pre-WC importance; may need regenerate for thesis roster |

---

## Reference configs

| File | Status | Feeds | Note |
|------|--------|-------|------|
| [docs/reference/daily_trigger.yml.reference](../reference/daily_trigger.yml.reference) | `reference` | system (Ch 4); appendix | Cloud Scheduler shape |
| [docs/reference/tournament_trigger.yml.reference](../reference/tournament_trigger.yml.reference) | `reference` | system (Ch 4); appendix | WC cadence shape |

---

## Archive

| File | Status | Feeds | Note |
|------|--------|-------|------|
| [docs/archive/feature_v2_baseline.md](../archive/feature_v2_baseline.md) | `stale` | journal; results §5.1 | Gold v2 champion snapshot 2026-04-29 |
| [docs/archive/repo_plan.md](../archive/repo_plan.md) | `stale` | journal | Early repo plan |
| [docs/archive/requirements.md](../archive/requirements.md) | `stale` | journal | Early requirements |
| [docs/archive/presentation/qa_prep.md](../archive/presentation/qa_prep.md) | `stale` | journal | Course presentation Q&A |
| [docs/archive/presentation/build_slides.py](../archive/presentation/build_slides.py) | `stale` | — | Slide builder |
| [docs/archive/presentation/wc_mlops_presentation_draft.bak.pptx](../archive/presentation/wc_mlops_presentation_draft.bak.pptx) | `stale` | — | Binary backup |

---

## Known gaps

1. **Literature on retraining cadence / concept drift / continuous training** is absent from `literature_review.md` and `references.bib` (16 entries, all football goal modeling or general MLOps). Needed before paper related-work / discussion.
2. **D.2** (structural-driver regression) and **D.3** (WC 2022 frozen retrospective) are open in `thesis.md`.
3. **Findings memo** (`docs/notes/findings_memo.md`) has RQ1 and RQ2 — RQ3 claim language still open.
4. **Stale LaTeX** (`paper.tex`, optionally `wc_mlops_report.tex`) not yet moved under `docs/archive/` — marked only; move is a follow-up.
5. **University thesis guidelines** (page/word limits) unchecked (D.4 item).
6. **New paper Overleaf file** does not exist; do not revise the course `paper.tex` in place.
