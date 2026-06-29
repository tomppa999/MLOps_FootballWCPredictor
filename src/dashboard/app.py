"""Streamlit dashboard for WC 2026 tournament simulations.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from src.dashboard.load_artifacts import (
        InferenceRunInfo,
        load_group_mapping,
        load_latest_inference_artifacts,
        load_latest_monitoring_results,
    )
except ModuleNotFoundError:
    from load_artifacts import (  # type: ignore
        InferenceRunInfo,
        load_group_mapping,
        load_latest_inference_artifacts,
        load_latest_monitoring_results,
    )

_KO_STAGE_ORDER = {"R32": 0, "R16": 1, "QF": 2, "SF": 3, "Final": 4}
_KO_STAGE_LABELS = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-finals",
                    "SF": "Semi-finals", "Final": "Final"}

# Determined pairings log a pairing_frequency of 1.0; treat near-1.0 values as
# locked too. 0.999 is inclusive, so 0.9989-style values stay "predicted".
_LOCKED_PAIRING_THRESHOLD = 0.999


def _format_percentage_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy with selected columns scaled to 0-100 and rounded."""
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = (out[col].astype(float) * 100).round(1)
    return out


def _render_run_metadata(info: InferenceRunInfo) -> None:
    st.sidebar.markdown("**Run metadata**")
    st.sidebar.text(f"run_id: {getattr(info, 'run_id', 'unknown')}")
    n_sims = getattr(info, "n_sims", None)
    if n_sims is not None:
        st.sidebar.text(f"n_sims: {n_sims}")
    champion_run_id = getattr(info, "champion_run_id", None)
    if champion_run_id:
        st.sidebar.text(f"champion_run_id: {champion_run_id}")
    inference_timestamp = getattr(info, "inference_timestamp", None)
    if inference_timestamp:
        st.sidebar.text(f"timestamp: {inference_timestamp}")


# ---------------------------------------------------------------------------
# View A: Tournament overview
# ---------------------------------------------------------------------------

def view_tournament_overview(tournament_df: pd.DataFrame) -> None:
    st.header("Tournament advancement probabilities")

    cols = ["p_r32", "p_r16", "p_qf", "p_sf", "p_final", "p_winner"]
    display_cols = ["R32", "R16", "QF", "SF", "Final", "Winner"]
    col_to_display = dict(zip(cols, display_cols))

    sort_options = {label: col for col, label in col_to_display.items()}
    sort_by_label = st.sidebar.selectbox(
        "Sort teams by",
        options=list(reversed(display_cols)),
        index=0,
    )
    sort_col = sort_options[sort_by_label]

    df = tournament_df.copy()
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df = _format_percentage_columns(df, cols)

    team_order = df["team"].tolist()
    pivot = df.set_index("team")[cols].rename(columns=col_to_display)
    pivot = pivot[display_cols]
    pivot = pivot.loc[team_order]

    fig = px.imshow(
        pivot,
        color_continuous_scale="Greens",
        labels={"color": "Probability (%)"},
        aspect="auto",
    )
    fig.update_layout(
        yaxis_title="Team",
        xaxis_title="Stage",
        height=max(800, len(df) * 18),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Cells show the probability (in %) of each team reaching at least each stage.")


# ---------------------------------------------------------------------------
# View B: Group positions
# ---------------------------------------------------------------------------

def view_group_positions(group_df: pd.DataFrame, team_to_group: dict[str, str]) -> None:
    st.header("Group finish probabilities")

    df = group_df.copy()
    df["group"] = df["team"].map(team_to_group)

    positions = ["p_1st", "p_2nd", "p_3rd_qualify", "p_3rd_elim", "p_4th"]
    labels = ["1st", "2nd", "3rd (Q)", "3rd (E)", "4th"]

    df_long = df.melt(
        id_vars=["team", "group"],
        value_vars=positions,
        var_name="position",
        value_name="prob",
    )
    df_long["position_label"] = df_long["position"].map(dict(zip(positions, labels)))

    groups = sorted(df_long["group"].dropna().unique())
    for group in groups:
        st.subheader(f"Group {group}")
        gdf = df_long[df_long["group"] == group].copy()

        order = (
            gdf[gdf["position"] == "p_1st"]
            .sort_values("prob", ascending=True)["team"]
            .tolist()
        )
        gdf["team"] = pd.Categorical(gdf["team"], categories=order, ordered=True)
        gdf = gdf.sort_values(["team", "position_label"])

        fig = px.bar(
            gdf,
            x="prob",
            y="team",
            color="position_label",
            orientation="h",
            barmode="stack",
            color_discrete_sequence=["#2ecc71", "#82e0aa", "#f9e79f", "#f0b27a", "#e74c3c"],
            labels={"prob": "Probability", "team": "Team", "position_label": "Finish"},
            category_orders={"position_label": labels},
        )
        fig.update_layout(
            xaxis=dict(tickformat=".0%", range=[0, 1]),
            yaxis=dict(categoryorder="array", categoryarray=order),
            height=max(300, 70 * len(order)),
            margin=dict(l=80, r=10, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# KO fixture helpers
# ---------------------------------------------------------------------------


def resolve_ko_fixtures(ko_fixtures_df: pd.DataFrame) -> list[dict]:
    """Return KO fixtures as a list of dicts, ordered by stage then match_num.

    Each dict has: match_num, stage, home_team, away_team, status,
    home_goals, away_goals, decided_by, pairing_frequency.
    """
    if ko_fixtures_df is None or ko_fixtures_df.empty:
        return []
    df = ko_fixtures_df.copy()
    df["_stage_order"] = df["stage"].map(_KO_STAGE_ORDER).fillna(99)
    df = df.sort_values(["_stage_order", "match_num"]).drop(columns=["_stage_order"])
    return df.to_dict(orient="records")


def _ko_is_locked(fix: dict) -> bool:
    """Return True when a KO slot is settled (played, or a determined pairing).

    A slot is locked once it carries a real score (``status == "locked"``) or
    once its pairing is effectively certain (``pairing_frequency`` at/above
    ``_LOCKED_PAIRING_THRESHOLD``; determined pairings log exactly 1.0).
    """
    if fix.get("status") == "locked":
        return True
    freq = fix.get("pairing_frequency")
    return freq is not None and float(freq) >= _LOCKED_PAIRING_THRESHOLD


def _next_round_stage(ko_fixtures: list[dict]) -> str | None:
    """Return the earliest KO stage (by ``_KO_STAGE_ORDER``) still to be played.

    A match counts as played only once its slot is locked with a real score
    (``status == "locked"``); determined-but-unplayed pairings (status
    ``"predicted"``, even at pairing_frequency 1.0) still count as not-yet-played
    so the upcoming round surfaces. Returns None when every KO match is played
    or there are no fixtures.
    """
    unplayed_stages = {
        fix["stage"] for fix in ko_fixtures if fix.get("status") != "locked"
    }
    if not unplayed_stages:
        return None
    return min(unplayed_stages, key=lambda s: _KO_STAGE_ORDER.get(s, 99))


# ---------------------------------------------------------------------------
# View C: Match predictions (actual tournament fixtures)
# ---------------------------------------------------------------------------

def _lookup_prediction(
    pred_df: pd.DataFrame,
    home: str,
    away: str,
) -> dict | None:
    """Find the prediction row for a fixture (checking both orderings)."""
    mask = (pred_df["home_team"] == home) & (pred_df["away_team"] == away)
    row = pred_df.loc[mask]
    if not row.empty:
        r = row.iloc[0]
        return {
            "lambda_h": round(float(r["lambda_h"]), 3),
            "lambda_a": round(float(r["lambda_a"]), 3),
            "p_home": round(float(r["p_home"]) * 100, 1),
            "p_draw": round(float(r["p_draw"]) * 100, 1),
            "p_away": round(float(r["p_away"]) * 100, 1),
        }
    mask_rev = (pred_df["home_team"] == away) & (pred_df["away_team"] == home)
    row_rev = pred_df.loc[mask_rev]
    if not row_rev.empty:
        r = row_rev.iloc[0]
        return {
            "lambda_h": round(float(r["lambda_a"]), 3),
            "lambda_a": round(float(r["lambda_h"]), 3),
            "p_home": round(float(r["p_away"]) * 100, 1),
            "p_draw": round(float(r["p_draw"]) * 100, 1),
            "p_away": round(float(r["p_home"]) * 100, 1),
        }
    return None


def _completed_results_lookup(
    monitoring_df: pd.DataFrame | None,
    champion_model_name: str | None,
) -> dict[frozenset, dict]:
    """Map frozenset({home, away}) -> pre-kickoff probs (%) + actual score.

    Filters to the champion model and cadence_mode=frozen. Returns an empty
    dict when monitoring data is absent (pre-tournament or outage).
    """
    if monitoring_df is None or monitoring_df.empty:
        return {}
    df = monitoring_df.copy()
    if "cadence_mode" in df.columns:
        df = df[df["cadence_mode"] == "frozen"]
    if champion_model_name and "model_name" in df.columns:
        df = df[df["model_name"] == champion_model_name]
    if df.empty:
        return {}

    lookup: dict[frozenset, dict] = {}
    for _, r in df.iterrows():
        key: frozenset = frozenset({r["home"], r["away"]})
        lookup[key] = {
            "m_home": r["home"],
            "actual_home_goals": int(r["actual_h"]),
            "actual_away_goals": int(r["actual_a"]),
            "p_home_pct": round(float(r["p_home"]) * 100, 1),
            "p_draw_pct": round(float(r["p_draw"]) * 100, 1),
            "p_away_pct": round(float(r["p_away"]) * 100, 1),
        }
    return lookup


def _apply_completed_result(row: dict, lookup: dict[frozenset, dict]) -> dict:
    """Overlay pre-kickoff probs + actual score onto a fixture record if played.

    For played matches: replaces the (post-match) probabilities with the
    leakage-safe pre-kickoff ones and records actual_h/actual_a.
    For upcoming matches: sets played=False, actual_h/actual_a=None.
    """
    rec = lookup.get(frozenset({row["home_team"], row["away_team"]}))
    if rec is None:
        row["played"] = False
        row["actual_h"] = None
        row["actual_a"] = None
        return row

    row["played"] = True
    row["p_draw"] = rec["p_draw_pct"]
    if rec["m_home"] == row["home_team"]:
        row["p_home"] = rec["p_home_pct"]
        row["p_away"] = rec["p_away_pct"]
        row["actual_h"] = rec["actual_home_goals"]
        row["actual_a"] = rec["actual_away_goals"]
    else:
        # Config home is monitoring's away — flip sides.
        row["p_home"] = rec["p_away_pct"]
        row["p_away"] = rec["p_home_pct"]
        row["actual_h"] = rec["actual_away_goals"]
        row["actual_a"] = rec["actual_home_goals"]
    return row


def _build_match_bar(fixtures_with_preds: pd.DataFrame) -> None:
    """Render a stacked horizontal bar chart for a set of fixtures."""
    import plotly.graph_objects as go

    labels: list[str] = []
    p_home_vals: list[float] = []
    p_draw_vals: list[float] = []
    p_away_vals: list[float] = []
    home_texts: list[str] = []
    draw_texts: list[str] = []
    away_texts: list[str] = []

    for _, r in fixtures_with_preds.iterrows():
        if r.get("played") and r.get("actual_h") is not None:
            label = (
                f"{r['home_team']} {int(r['actual_h'])}–{int(r['actual_a'])} "
                f"{r['away_team']}  ·  pre-match odds"
            )
        else:
            label = f"{r['home_team']}  vs  {r['away_team']}"
        labels.append(label)
        ph = r.get("p_home", 0.0)
        pd_ = r.get("p_draw", 0.0)
        pa = r.get("p_away", 0.0)
        p_home_vals.append(ph)
        p_draw_vals.append(pd_)
        p_away_vals.append(pa)
        home_texts.append(f"{r['home_team']} {ph:.1f}%")
        draw_texts.append(f"Draw {pd_:.1f}%")
        away_texts.append(f"{r['away_team']} {pa:.1f}%")

    labels.reverse()
    p_home_vals.reverse()
    p_draw_vals.reverse()
    p_away_vals.reverse()
    home_texts.reverse()
    draw_texts.reverse()
    away_texts.reverse()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=p_home_vals,
        name="Home win",
        orientation="h",
        marker_color="#3498db",
        text=home_texts,
        textposition="inside",
        insidetextanchor="middle",
    ))
    fig.add_trace(go.Bar(
        y=labels,
        x=p_draw_vals,
        name="Draw",
        orientation="h",
        marker_color="#bdc3c7",
        text=draw_texts,
        textposition="inside",
        insidetextanchor="middle",
    ))
    fig.add_trace(go.Bar(
        y=labels,
        x=p_away_vals,
        name="Away win",
        orientation="h",
        marker_color="#e74c3c",
        text=away_texts,
        textposition="inside",
        insidetextanchor="middle",
    ))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(range=[0, 100], title="Probability (%)", showticklabels=False),
        yaxis=dict(title=""),
        height=max(200, len(labels) * 60),
        margin=dict(l=180, r=10, t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    st.plotly_chart(fig, use_container_width=True)


def view_match_predictions(
    pred_df: pd.DataFrame,
    ko_fixtures_df: pd.DataFrame | None = None,
    monitoring_df: pd.DataFrame | None = None,
    champion_model_name: str | None = None,
) -> None:
    st.header("Match predictions")
    st.caption(
        "Played matches show **pre-kickoff** probabilities "
        "(last inference run before kick-off) alongside the final score. "
        "Upcoming matches show the latest live prediction."
    )

    completed = _completed_results_lookup(monitoring_df, champion_model_name)
    ko_fixtures = resolve_ko_fixtures(ko_fixtures_df)
    next_stage = _next_round_stage(ko_fixtures)

    if not ko_fixtures:
        st.info("Knockout fixtures are not available yet.")
    elif next_stage is None:
        st.success("All knockout matches have been played — the tournament is complete.")
    else:
        st.subheader(f"Next round: {_KO_STAGE_LABELS.get(next_stage, next_stage)}")
        next_records: list[dict] = []
        for fix in ko_fixtures:
            if fix["stage"] != next_stage:
                continue
            pred = _lookup_prediction(pred_df, fix["home_team"], fix["away_team"])
            row = {
                "home_team": fix["home_team"],
                "away_team": fix["away_team"],
                "lambda_h": pred["lambda_h"] if pred else None,
                "lambda_a": pred["lambda_a"] if pred else None,
                "p_home": pred["p_home"] if pred else 0.0,
                "p_draw": pred["p_draw"] if pred else 0.0,
                "p_away": pred["p_away"] if pred else 0.0,
            }
            next_records.append(_apply_completed_result(row, completed))
        _build_match_bar(pd.DataFrame(next_records))

    # KO fixtures section
    if ko_fixtures:
        st.divider()
        st.subheader("Knockout stage")
        stage_groups: dict[str, list[dict]] = {}
        for fix in ko_fixtures:
            stage_groups.setdefault(fix["stage"], []).append(fix)
        for stage_key in sorted(stage_groups.keys(), key=lambda s: _KO_STAGE_ORDER.get(s, 99)):
            label = _KO_STAGE_LABELS.get(stage_key, stage_key)
            st.markdown(f"**{label}**")
            ko_records = []
            for fix in stage_groups[stage_key]:
                pred = _lookup_prediction(pred_df, fix["home_team"], fix["away_team"])
                status_badge = (
                    "🔒 Locked" if _ko_is_locked(fix)
                    else f"🔮 Predicted ({fix['pairing_frequency']:.0%})"
                )
                h, a = fix.get("home_goals"), fix.get("away_goals")
                score = (
                    f"{int(h)}–{int(a)}"
                    if fix.get("status") == "locked" and pd.notna(h) and pd.notna(a)
                    else ""
                )
                ko_records.append({
                    "status": status_badge,
                    "home_team": fix["home_team"],
                    "away_team": fix["away_team"],
                    "score": score,
                    "lambda_h": pred["lambda_h"] if pred else None,
                    "lambda_a": pred["lambda_a"] if pred else None,
                    "p_home": round(pred["p_home"] if pred else 0.0, 1),
                    "p_draw": round(pred["p_draw"] if pred else 0.0, 1),
                    "p_away": round(pred["p_away"] if pred else 0.0, 1),
                })
            ko_df = pd.DataFrame(ko_records)
            ko_df.columns = ["Status", "Home", "Away", "Score", "xG Home", "xG Away",
                              "P(H) %", "P(D) %", "P(A) %"]
            st.dataframe(ko_df, use_container_width=True)
        st.caption(
            "Predicted matchups show the most likely pairing for each slot based on "
            "independent marginal probabilities from the latest simulation run. "
            "These are per-slot estimates and do not represent a single coherent bracket path."
        )


# ---------------------------------------------------------------------------
# View D: Most common matchups
# ---------------------------------------------------------------------------

def view_common_matchups(ko_df: pd.DataFrame) -> None:
    st.header("Most common matchups")

    stages_available = sorted(ko_df["stage"].unique())
    selected_stages = st.sidebar.multiselect(
        "Filter by stage",
        options=stages_available,
        default=stages_available,
    )
    top_n = st.sidebar.slider("Top N matchups", min_value=5, max_value=100, value=20, step=5)

    df = ko_df.copy()
    if selected_stages:
        df = df[df["stage"].isin(selected_stages)]

    df = df.sort_values("frequency", ascending=False).head(top_n)
    df["matchup"] = df["team_a"] + " vs " + df["team_b"]
    df["freq_pct"] = (df["frequency"] * 100).round(1)

    fig = px.bar(
        df,
        x="freq_pct",
        y="matchup",
        color="stage",
        orientation="h",
        labels={"freq_pct": "Frequency (%)", "matchup": "Matchup", "stage": "Stage"},
    )
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        height=max(400, top_n * 28),
        margin=dict(l=160, r=10, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="WC 2026 Predictions", layout="wide")
    st.title("WC 2026 Simulation Dashboard")

    try:
        data, info = load_latest_inference_artifacts()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load latest inference artifacts: {exc}")
        return

    # Guard against stale @st.cache_data pickles surviving a hot-reload: if the
    # cached object's class identity changed, clear the cache and rerun silently.
    try:
        is_stale = info.is_stale
    except AttributeError:
        load_latest_inference_artifacts.clear()
        st.rerun()
        return

    if is_stale:
        ts = getattr(info, "inference_timestamp", None)
        st.warning(
            "Live tracking server (DagsHub) is currently unreachable — "
            "showing the last cached snapshot"
            + (f" from {ts}" if ts else "")
            + ". Predictions auto-refresh within 5 minutes once it's back."
        )

    _render_run_metadata(info)

    tournament_df = data.get("tournament_probabilities")
    group_df = data.get("group_positions")
    pred_df = data.get("predictions")
    ko_df = data.get("ko_pairings")
    ko_fixtures_df = data.get("ko_fixtures")
    monitoring_df = load_latest_monitoring_results()

    if group_df is not None:
        team_to_group = load_group_mapping()
    else:
        team_to_group = {}

    views = [
        "Tournament overview",
        "Group positions",
        "Match predictions",
        "Common matchups",
    ]
    view = st.sidebar.radio("View", options=views)

    if view == "Tournament overview":
        if tournament_df is None:
            st.warning("tournament_probabilities.csv not found.")
        else:
            view_tournament_overview(tournament_df)
    elif view == "Group positions":
        if group_df is None:
            st.warning("group_positions.csv not found.")
        else:
            view_group_positions(group_df, team_to_group)
    elif view == "Match predictions":
        if pred_df is None:
            st.warning("predictions.csv not found.")
        else:
            view_match_predictions(
                pred_df,
                ko_fixtures_df=ko_fixtures_df,
                monitoring_df=monitoring_df,
                champion_model_name=getattr(info, "champion_model_name", None),
            )
    else:
        if ko_df is None:
            st.warning("ko_pairings.csv not found.")
        else:
            view_common_matchups(ko_df)


if __name__ == "__main__":
    main()
