"""Streamlit dashboard for WC 2026 tournament simulations.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from src.dashboard.load_artifacts import (
        InferenceRunInfo,
        load_group_mapping,
        load_latest_inference_artifacts,
    )
except ModuleNotFoundError:
    from load_artifacts import (  # type: ignore
        InferenceRunInfo,
        load_group_mapping,
        load_latest_inference_artifacts,
    )

TOURNAMENT_CONFIG_PATH = Path("data/tournament/wc2026.json")


def _format_percentage_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy with selected columns scaled to 0-100 and rounded."""
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = (out[col].astype(float) * 100).round(1)
    return out


def _render_run_metadata(info: InferenceRunInfo) -> None:
    st.sidebar.markdown("**Run metadata**")
    st.sidebar.text(f"run_id: {info.run_id}")
    if info.n_sims is not None:
        st.sidebar.text(f"n_sims: {info.n_sims}")
    if info.champion_run_id:
        st.sidebar.text(f"champion_run_id: {info.champion_run_id}")
    if info.inference_timestamp:
        st.sidebar.text(f"timestamp: {info.inference_timestamp}")


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
# View C: Match predictions (actual tournament fixtures)
# ---------------------------------------------------------------------------

def _load_tournament_fixtures() -> pd.DataFrame:
    """Build a DataFrame of actual group-stage fixtures from wc2026.json.

    Returns columns: group, matchday, home_team, away_team.
    """
    with TOURNAMENT_CONFIG_PATH.open() as f:
        config = json.load(f)

    groups: dict[str, list[str]] = config["groups"]
    matchdays_cfg = config.get("group_matchdays", [])
    if not matchdays_cfg:
        matchdays_cfg = [
            {"matchday": 1, "pairs": [[0, 1], [2, 3]]},
            {"matchday": 2, "pairs": [[0, 2], [3, 1]]},
            {"matchday": 3, "pairs": [[3, 0], [1, 2]]},
        ]

    rows: list[dict] = []
    for group_letter, teams in groups.items():
        for md in matchdays_cfg:
            for h_idx, a_idx in md["pairs"]:
                rows.append({
                    "group": group_letter,
                    "matchday": md["matchday"],
                    "home_team": teams[h_idx],
                    "away_team": teams[a_idx],
                })
    return pd.DataFrame(rows)


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


def view_match_predictions(pred_df: pd.DataFrame) -> None:
    st.header("Match predictions")

    fixtures = _load_tournament_fixtures()

    records: list[dict] = []
    for _, fix in fixtures.iterrows():
        pred = _lookup_prediction(pred_df, fix["home_team"], fix["away_team"])
        row = {
            "group": fix["group"],
            "matchday": fix["matchday"],
            "home_team": fix["home_team"],
            "away_team": fix["away_team"],
            "lambda_h": pred["lambda_h"] if pred else None,
            "lambda_a": pred["lambda_a"] if pred else None,
            "p_home": pred["p_home"] if pred else 0.0,
            "p_draw": pred["p_draw"] if pred else 0.0,
            "p_away": pred["p_away"] if pred else 0.0,
        }
        records.append(row)

    result_df = pd.DataFrame(records)

    matchdays = sorted(result_df["matchday"].unique())
    md_options = ["All"] + [f"MD {int(m)}" for m in matchdays]
    selected_md_label = st.radio("Matchday", options=md_options, horizontal=True)

    groups = sorted(result_df["group"].unique())
    selected_group = st.sidebar.selectbox("Filter by group", options=["All"] + groups)

    display_df = result_df.copy()
    if selected_md_label != "All":
        md_num = int(selected_md_label.split()[-1])
        display_df = display_df[display_df["matchday"] == md_num]
    if selected_group != "All":
        display_df = display_df[display_df["group"] == selected_group]

    display_df = display_df.sort_values(["group", "matchday"]).reset_index(drop=True)

    for group in sorted(display_df["group"].unique()):
        st.subheader(f"Group {group}")
        gdf = display_df[display_df["group"] == group]
        _build_match_bar(gdf)

    with st.expander("Raw data"):
        raw = display_df.copy()
        raw.columns = ["Group", "MD", "Home", "Away", "xG Home", "xG Away",
                        "P(H) %", "P(D) %", "P(A) %"]
        st.dataframe(raw, use_container_width=True)


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

    _render_run_metadata(info)

    tournament_df = data.get("tournament_probabilities")
    group_df = data.get("group_positions")
    pred_df = data.get("predictions")
    ko_df = data.get("ko_pairings")

    if group_df is not None:
        team_to_group = load_group_mapping()
    else:
        team_to_group = {}

    views = ["Tournament overview", "Group positions", "Match predictions", "Common matchups"]
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
            view_match_predictions(pred_df)
    else:
        if ko_df is None:
            st.warning("ko_pairings.csv not found.")
        else:
            view_common_matchups(ko_df)


if __name__ == "__main__":
    main()
