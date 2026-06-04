"""
Build the WC MLOps presentation draft from the TAU template.
Run from the project root:
    python3.12 docs/presentation/build_slides.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.pptx_deps"))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

# ── Colours ──────────────────────────────────────────────────────────────────
TAU_PURPLE  = RGBColor(0x5C, 0x16, 0x7E)   # approx TAU brand purple
TAU_WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK_GRAY   = RGBColor(0x44, 0x44, 0x44)
LIGHT_GRAY  = RGBColor(0xF0, 0xF0, 0xF0)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.join(os.path.dirname(__file__), "../..")
TEMPLATE   = os.path.join(ROOT, "TAU template EN.pptx")
FIG_ARCH   = os.path.join(ROOT, "docs/figures/Architecture.png")
FIG_CT     = os.path.join(ROOT, "docs/figures/architecture_ct_pipeline.png")
OUT        = os.path.join(ROOT, "docs/presentation/wc_mlops_presentation_draft.pptx")


# ── Helpers ───────────────────────────────────────────────────────────────────
def delete_slide(prs, index: int):
    """Remove slide at position *index* from the presentation."""
    xml_slides = prs.slides._sldIdLst
    slide_id_elem = xml_slides[index]
    r_id = slide_id_elem.get(
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    )
    xml_slides.remove(slide_id_elem)
    prs.slides.part.drop_rel(r_id)


def add_slide(prs, layout_idx: int):
    return prs.slides.add_slide(prs.slide_layouts[layout_idx])


def bullets(tf, items, size=15):
    """
    Fill *tf* (TextFrame) with bulleted paragraphs.
    *items* = list of (level, text) tuples. level 0 = top, 1 = sub-bullet.
    """
    tf.clear()
    first = True
    for level, text in items:
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.level = level
        run = para.add_run()
        run.text = text
        run.font.size = Pt(size)


def add_table(slide, data, left, top, width, height,
              header_bg=TAU_PURPLE, row_alt=LIGHT_GRAY):
    """Add a table to a slide; first row is treated as header."""
    rows, cols = len(data), len(data[0])
    tbl = slide.shapes.add_table(rows, cols, left, top, width, height).table
    for r, row in enumerate(data):
        for c, cell_text in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = cell_text
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.CENTER
            run = para.runs[0] if para.runs else para.add_run()
            run.font.size = Pt(12)
            run.font.bold = (r == 0)
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_bg
                run.font.color.rgb = TAU_WHITE
            elif r % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = row_alt
    return tbl


def caption(slide, text, left, top, width, height, size=11):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.italic = True
    run.font.color.rgb = DARK_GRAY
    return txb


# ── Open template & clear example slides ─────────────────────────────────────
prs = Presentation(TEMPLATE)
# Template ships with 4 example slides — remove them all (back-to-front)
for _ in range(len(prs.slides) - 1, -1, -1):
    delete_slide(prs, _)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1 — Title / Cover
#   Layout 1: "Title Slide - purple"  (idx 0 = title, idx 1 = subtitle)
# ─────────────────────────────────────────────────────────────────────────────
s = add_slide(prs, 1)
for ph in s.placeholders:
    if ph.placeholder_format.idx == 0:
        ph.text = "MLOps for Football\nWorld Cup Prediction"
    elif ph.placeholder_format.idx == 1:
        ph.text = (
            "Multi-Model Comparison with Structured Model Promotion\n\n"
            "Tom Farnschläder  ·  Tampere University  ·  May 2026"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2 — Motivation
#   Layout 3: "Title and Content - white"
# ─────────────────────────────────────────────────────────────────────────────
s = add_slide(prs, 3)
for ph in s.placeholders:
    if ph.placeholder_format.idx == 0:
        ph.text = "Motivation"
    elif ph.placeholder_format.idx == 1:
        bullets(ph.text_frame, [
            (0, "Who wins the 2026 World Cup? And can we build a rigorous ML system to answer that?"),
            (0, "National team football is hard to predict:"),
            (1, "Only 6–12 matches per year → sparse, irregular data"),
            (1, "Cross-confederation heterogeneity; friendlies vs. finals mix in training data"),
            (1, "High variance in a low-scoring sport"),
            (0, "2026: expanded 48-team format with best-third-place rule → combinatorial bracket uncertainty"),
            (0, "Two contributions:"),
            (1, "A practical MLOps pipeline: Bronze/Silver/Gold + three-environment model lifecycle"),
            (1, "Systematic comparison of nine candidate models spanning five families"),
        ], size=16)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3 — System Overview (architecture diagram)
#   Layout 9: "Title Only - white"
# ─────────────────────────────────────────────────────────────────────────────
s = add_slide(prs, 9)
for ph in s.placeholders:
    if ph.placeholder_format.idx == 0:
        ph.text = "End-to-End MLOps Pipeline"

s.shapes.add_picture(FIG_ARCH, Inches(0.4), Inches(1.35), Inches(12.5), Inches(5.75))


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4 — Data Pipeline & Features
#   Layout 3: "Title and Content - white"
# ─────────────────────────────────────────────────────────────────────────────
s = add_slide(prs, 3)
for ph in s.placeholders:
    if ph.placeholder_format.idx == 0:
        ph.text = "Data Pipeline & Feature Engineering"
    elif ph.placeholder_format.idx == 1:
        bullets(ph.text_frame, [
            (0, "Bronze: raw immutable snapshots — API-Football (fixtures + per-match stats) and Elo ratings (eloratings.net)"),
            (0, "Silver: cleaned Parquet — schema validation, team name mapping, competition tier (1–4), Elo join (date-proximate)"),
            (0, "Gold: one row per match, strictly time-aware features (no future information)"),
            (0, "Feature families:"),
            (1, "Elo: elo_diff (favourite signal) + elo_sum (match quality, added in Gold v2)"),
            (1, "Rolling form (last 10 matches): goals for/against, shot volume/precision, fouls, corners, possession %"),
            (1, "Context: competition_tier, is_knockout, is_neutral"),
            (1, "Squad cohesion: days_since_last_match, rest_diff (Gold v2)"),
            (0, "Clustering experiment: silhouette < 0.40 across all k/variants → no discrete tactical archetypes → raw rolling columns used directly"),
            (0, "~51% stats coverage → fine-grained features (e.g. shots-inside-box: ~1.6% non-null) excluded from core feature set"),
        ], size=14)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5 — Nine Candidate Models & Experimental Setup
#   Layout 3: "Title and Content - white"
# ─────────────────────────────────────────────────────────────────────────────
s = add_slide(prs, 3)
for ph in s.placeholders:
    if ph.placeholder_format.idx == 0:
        ph.text = "Nine Candidate Models & Experimental Setup"
    elif ph.placeholder_format.idx == 1:
        bullets(ph.text_frame, [
            (0, "Modeling objective: predict goal-rate distributions P(home, away) — W/D/L derived downstream via score marginalization + Monte Carlo simulation"),
            (0, "Nine candidates across five families:"),
            (1, "Statistical: Poisson GLM (bivariate Karlis-Ntzoufras), NegBin GLM"),
            (1, "Bayesian: Bayesian Poisson (PyMC / ADVI)"),
            (1, "Time-series: SARIMAX (integer match-index time axis)"),
            (1, "ML: Ridge, Random Forest, XGBoost"),
            (1, "Deep learning: LSTM, 1D CNN (Keras / JAX)"),
            (0, "Data split: pre-WC 2022 training (3,903 rows) | WC 2022 holdout (64 matches, never seen during training)"),
            (0, "Tuning: Optuna TPE, 50 trials, walk-forward expanding CV; objective = Poisson NLL (goal-count space)"),
            (0, "QA metric: Ranked Probability Score (RPS) on WC 2022 holdout — lower is better"),
            (0, "All nine advance to QA (no top-K gate — empirically motivated; see next slide)"),
        ], size=14)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 6 — Finding 1: CV–Holdout Ranking Reversal
#   Layout 9: "Title Only - white" + two tables
# ─────────────────────────────────────────────────────────────────────────────
s = add_slide(prs, 9)
for ph in s.placeholders:
    if ph.placeholder_format.idx == 0:
        ph.text = "Finding 1: CV–Holdout Ranking Reversal"

cv_data = [
    ["Rank", "Model", "CV NLL"],
    ["1", "NegBin GLM", "2.758"],
    ["2", "Bayesian Poisson", "2.759"],
    ["3", "Poisson GLM", "2.763"],
    ["4 ★", "XGBoost", "2.767"],
    ["5", "Random Forest", "2.768"],
    ["6", "LSTM", "2.821"],
    ["7", "Ridge", "2.847"],
    ["8", "SARIMAX", "3.016"],
    ["9", "1D CNN", "3.039"],
]
qa_data = [
    ["Rank", "Model", "Holdout RPS"],
    ["1 ★", "XGBoost", "0.2109"],
    ["2", "Random Forest", "0.2121"],
    ["3", "Ridge", "0.2141"],
    ["4", "Poisson GLM", "0.2159"],
    ["5", "Bayesian Poisson", "0.2166"],
    ["6", "NegBin GLM", "0.2170"],
    ["7", "SARIMAX", "0.2183"],
    ["8", "LSTM", "0.2207"],
    ["9", "1D CNN", "0.2248"],
]

add_table(s, cv_data,
          Inches(0.3), Inches(1.35), Inches(5.9), Inches(4.8))
add_table(s, qa_data,
          Inches(6.5), Inches(1.35), Inches(6.5), Inches(4.8))

caption(s,
    "★ XGBoost: rank 4 in CV NLL → rank 1 in holdout RPS.  "
    "CV winner (NegBin GLM) drops to rank 6.  "
    "Two explanations: (1) metric mismatch — NLL rewards scoreline accuracy; RPS evaluates W/D/L probabilities.  "
    "(2) distribution shift — WC holdout is tournament-only (neutral venues, strong teams); training set contains many tier-3/4 matches.  "
    "→ Design decision: removed top-4 promotion gate; all nine models advance to QA.",
    Inches(0.3), Inches(6.3), Inches(12.7), Inches(1.0))


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 7 — Finding 2: Elo Dominates
#   Layout 9: "Title Only - white" + feature importance table + notes
# ─────────────────────────────────────────────────────────────────────────────
s = add_slide(prs, 9)
for ph in s.placeholders:
    if ph.placeholder_format.idx == 0:
        ph.text = "Finding 2: Elo Differential Dominates All Models"

fi_data = [
    ["Model", "#1 Feature (NLL Δ)", "#2 Feature (NLL Δ)", "#3 Feature (NLL Δ)"],
    ["Random Forest",     "elo_diff (+.121)",   "elo_sum (+.002)",         "roll_goals_ag_h (+.002)"],
    ["SARIMAX",           "elo_diff (+.110)",   "roll_goals_ag_h (+.002)", "elo_sum (+.000)"],
    ["NegBin GLM",        "elo_diff (+.104)",   "roll_goals_ag_h (+.001)", "comp_tier (+.001)"],
    ["Bayesian Poisson",  "elo_diff (+.103)",   "rest_diff (+.001)",       "roll_goals_ag_h (+.001)"],
    ["Poisson GLM",       "elo_diff (+.102)",   "roll_goals_ag_h (+.001)", "comp_tier (+.001)"],
    ["XGBoost",           "elo_diff (+.098)",   "roll_tac_poss_a (+.002)", "roll_goals_ag_h (+.002)"],
    ["LSTM",              "elo_diff (+.095)",   "is_neutral (+.005)",      "comp_tier (+.002)"],
    ["Ridge",             "elo_diff (+.062)",   "roll_goals_ag_h (+.004)", "roll_goals_ag_a (+.002)"],
    ["1D CNN",            "elo_diff (+.055)",   "is_neutral (+.005)",      "comp_tier (+.004)"],
]

add_table(s, fi_data,
          Inches(0.3), Inches(1.35), Inches(12.7), Inches(4.5))

caption(s,
    "elo_diff ranks #1 across ALL nine model families — importance 1–2 orders of magnitude above any other feature.  "
    "Narrow holdout RPS band: 0.010 across 8 of 9 models → shared feature ceiling limits gains from more complex architectures.  "
    "Gold v2 auto-promotion: XGBoost ΔRPS = −0.0009 over v1 → registered as wc_production version 4.  "
    "Deep learning penalty confirmed (H4): LSTM degraded +0.0074 after adding 5 features to an already data-sparse corpus.",
    Inches(0.3), Inches(6.1), Inches(12.7), Inches(1.1))


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 8 — Live Demo
#   Layout 4: "Title and Content - purple"
# ─────────────────────────────────────────────────────────────────────────────
s = add_slide(prs, 4)
for ph in s.placeholders:
    if ph.placeholder_format.idx == 0:
        ph.text = "Live Demo"
    elif ph.placeholder_format.idx == 1:
        bullets(ph.text_frame, [
            (0, "MLflow UI"),
            (1, "Experiment runs and Optuna trial audit trail"),
            (1, "wc_production champion alias  |  wc_staging WC 2022 audit trail  |  wc_shadow monitoring candidates"),
            (0, "Tournament simulation output"),
            (1, "Per-team group-stage advancement and knockout probabilities"),
            (1, "10,000 Monte Carlo paths; 60%+ of knockout matchups non-deterministic under best-third-place rule"),
            (0, "Inference cycle walk-through"),
            (1, "Trigger → freshness check (ELO SHA-256 + API-Football 48h window)"),
            (1, "→ dvc repro (Silver / Gold rebuild with updated rolling features)"),
            (1, "→ champion predict → simulation artifacts logged to MLflow"),
        ], size=17)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 9 — Summary / Conclusion
#   Layout 3: "Title and Content - white"
# ─────────────────────────────────────────────────────────────────────────────
s = add_slide(prs, 3)
for ph in s.placeholders:
    if ph.placeholder_format.idx == 0:
        ph.text = "Summary"
    elif ph.placeholder_format.idx == 1:
        bullets(ph.text_frame, [
            (0, "XGBoost selected as production champion: holdout RPS 0.2109 on 64 WC 2022 matches"),
            (0, "CV–holdout ranking reversal motivated removing the top-K gate → all nine advance to QA"),
            (0, "Elo differential is the dominant signal; the shared feature ceiling limits gains from more complex architectures"),
            (0, "MLOps iteration loop works: Gold v2 auto-promoted XGBoost (ΔRPS −0.0009) without manual intervention"),
            (0, "Next: WC 2026 live monitoring — per-match scoring of all nine models against actual results every 30 minutes"),
            (0, "Limitations:"),
            (1, "Single holdout (89 matches) — limited statistical power to separate closely ranked models"),
            (1, "WC 2022 holdout is 4 years from prediction target; tactical/generational drift unmodeled"),
        ], size=16)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 10 — Ending
#   Layout 32: "Ending EN - purple"
# ─────────────────────────────────────────────────────────────────────────────
add_slide(prs, 32)


# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT), exist_ok=True)
prs.save(OUT)
print(f"Saved → {OUT}")
