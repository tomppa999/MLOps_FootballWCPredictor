import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.pipeline.trigger import (
    VALID_MODES,
    _find_latest_run_manifest,
    _last_per_round_refit_matchday,
    _load_manifest_hashes,
    _load_slugs,
    _parse_args,
    _try_acquire_lock,
    check_api_football_freshness,
    check_elo_freshness,
    dispatch_training_or_inference,
    main,
    run_dvc_pipeline,
    run_elo_ingestion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_elo_manifest(path: Path, entries: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["elo_slug,source_url,downloaded_at_utc,status,http_status,file_name,file_sha256,notes"]
    for slug, sha in entries.items():
        lines.append(f"{slug},url,2026-01-01,downloaded,200,{slug}.tsv,{sha},")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_mapping_csv(path: Path, slugs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["team_name,elo_slug"] + [f"Team_{s},{s}" for s in slugs]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_run_manifest(runs_dir: Path, kept: int, run_id: str = "2026-03-31T04-30-00Z") -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "source": "api_football",
        "requested_window": {},
        "started_at_utc": "2026-03-31T04:30:00+00:00",
        "finished_at_utc": "2026-03-31T04:31:00+00:00",
        "summary": {"kept_fixture_count_total": kept},
    }
    (runs_dir / f"{run_id}_run_manifest.json").write_text(json.dumps(manifest))


def _write_fixtures(fixtures_dir: Path, statuses: list[str], window: str = "2026-03-29_2026-03-31") -> None:
    window_dir = fixtures_dir / window
    window_dir.mkdir(parents=True, exist_ok=True)
    response = [
        {"fixture": {"id": i, "status": {"short": s, "long": ""}}}
        for i, s in enumerate(statuses, 1)
    ]
    (window_dir / "fixtures.json").write_text(json.dumps({"response": response}))


# ---------------------------------------------------------------------------
# _load_manifest_hashes
# ---------------------------------------------------------------------------

def test_load_manifest_hashes_returns_empty_for_missing_file(tmp_path):
    assert _load_manifest_hashes(tmp_path / "missing.csv") == {}


def test_load_manifest_hashes_reads_slug_and_hash(tmp_path):
    f = tmp_path / "manifest.csv"
    _write_elo_manifest(f, {"France": "abc123", "Germany": "def456"})
    result = _load_manifest_hashes(f)
    assert result == {"France": "abc123", "Germany": "def456"}


# ---------------------------------------------------------------------------
# _load_slugs
# ---------------------------------------------------------------------------

def test_load_slugs_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_slugs(tmp_path / "missing.csv")


def test_load_slugs_returns_sorted_unique_slugs(tmp_path):
    f = tmp_path / "mapping.csv"
    _write_mapping_csv(f, ["Germany", "France", "Germany"])
    result = _load_slugs(f)
    assert result == ["France", "Germany"]


def test_load_slugs_raises_for_missing_column(tmp_path):
    f = tmp_path / "mapping.csv"
    f.write_text("team_name,other_col\nGermany,x\n")
    with pytest.raises(ValueError, match="elo_slug"):
        _load_slugs(f)


# ---------------------------------------------------------------------------
# _find_latest_run_manifest
# ---------------------------------------------------------------------------

def test_find_latest_run_manifest_returns_none_for_missing_dir(tmp_path):
    assert _find_latest_run_manifest(tmp_path / "runs") is None


def test_find_latest_run_manifest_returns_none_when_empty(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    assert _find_latest_run_manifest(runs) is None


def test_find_latest_run_manifest_returns_most_recent(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "2026-03-29T00-00-00Z_run_manifest.json").write_text("{}")
    (runs / "2026-03-31T04-30-00Z_run_manifest.json").write_text("{}")
    result = _find_latest_run_manifest(runs)
    assert result.name == "2026-03-31T04-30-00Z_run_manifest.json"


# ---------------------------------------------------------------------------
# check_elo_freshness
# ---------------------------------------------------------------------------

def test_elo_freshness_true_when_no_manifest(tmp_path):
    result = check_elo_freshness(
        manifest_file=tmp_path / "missing.csv",
        mapping_file=tmp_path / "mapping.csv",
    )
    assert result is True


def test_elo_freshness_false_when_load_slugs_fails(tmp_path, mocker):
    manifest = tmp_path / "manifest.csv"
    _write_elo_manifest(manifest, {"France": "abc"})
    mocker.patch("src.pipeline.trigger._load_slugs", side_effect=FileNotFoundError)
    result = check_elo_freshness(manifest_file=manifest, mapping_file=tmp_path / "mapping.csv")
    assert result is False


def test_elo_freshness_false_when_all_hashes_unchanged(tmp_path, mocker):
    manifest = tmp_path / "manifest.csv"
    _write_elo_manifest(manifest, {"France": "hash_fr", "Germany": "hash_de"})
    mapping = tmp_path / "mapping.csv"
    _write_mapping_csv(mapping, ["France", "Germany"])
    mocker.patch("src.pipeline.trigger._download_to", return_value=(True, "ok"))
    mocker.patch("src.pipeline.trigger._sha256", side_effect=["hash_fr", "hash_de"])
    result = check_elo_freshness(manifest_file=manifest, mapping_file=mapping)
    assert result is False


def test_elo_freshness_true_when_one_hash_changed(tmp_path, mocker):
    manifest = tmp_path / "manifest.csv"
    _write_elo_manifest(manifest, {"France": "old_hash", "Germany": "hash_de"})
    mapping = tmp_path / "mapping.csv"
    _write_mapping_csv(mapping, ["France", "Germany"])
    mocker.patch("src.pipeline.trigger._download_to", return_value=(True, "ok"))
    mocker.patch("src.pipeline.trigger._sha256", side_effect=["new_hash", "hash_de"])
    result = check_elo_freshness(manifest_file=manifest, mapping_file=mapping)
    assert result is True


def test_elo_freshness_false_when_all_downloads_fail(tmp_path, mocker):
    manifest = tmp_path / "manifest.csv"
    _write_elo_manifest(manifest, {"France": "abc"})
    mapping = tmp_path / "mapping.csv"
    _write_mapping_csv(mapping, ["France"])
    mocker.patch("src.pipeline.trigger._download_to", return_value=(False, "503 error"))
    result = check_elo_freshness(manifest_file=manifest, mapping_file=mapping)
    assert result is False


# ---------------------------------------------------------------------------
# check_api_football_freshness
# ---------------------------------------------------------------------------

def _mock_subprocess_ok(mocker) -> MagicMock:
    return mocker.patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="", stdout=""))


def test_api_freshness_false_on_ingestion_failure(tmp_path, mocker):
    mocker.patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="fail", stdout=""))
    result = check_api_football_freshness(
        runs_dir=tmp_path / "runs",
        fixtures_dir=tmp_path / "fixtures",
    )
    assert result is False


def test_api_freshness_false_when_no_manifest(tmp_path, mocker):
    _mock_subprocess_ok(mocker)
    result = check_api_football_freshness(
        runs_dir=tmp_path / "runs",
        fixtures_dir=tmp_path / "fixtures",
    )
    assert result is False


def test_api_freshness_false_when_no_fixtures_in_window(tmp_path, mocker):
    _mock_subprocess_ok(mocker)
    _write_run_manifest(tmp_path / "runs", kept=0)
    result = check_api_football_freshness(
        runs_dir=tmp_path / "runs",
        fixtures_dir=tmp_path / "fixtures",
    )
    assert result is False


def test_api_freshness_true_when_finished_and_in_progress(tmp_path, mocker):
    """In-progress games no longer block processing of finished ones."""
    _mock_subprocess_ok(mocker)
    _write_run_manifest(tmp_path / "runs", kept=2)
    _write_fixtures(tmp_path / "fixtures", statuses=["HT", "FT"])
    result = check_api_football_freshness(
        runs_dir=tmp_path / "runs",
        fixtures_dir=tmp_path / "fixtures",
    )
    assert result is True


def test_api_freshness_false_when_only_in_progress(tmp_path, mocker):
    """All in-progress and none finished → still False (no settled data to process)."""
    _mock_subprocess_ok(mocker)
    _write_run_manifest(tmp_path / "runs", kept=2)
    _write_fixtures(tmp_path / "fixtures", statuses=["HT", "1H"])
    result = check_api_football_freshness(
        runs_dir=tmp_path / "runs",
        fixtures_dir=tmp_path / "fixtures",
    )
    assert result is False


def test_api_freshness_false_when_only_cancelled(tmp_path, mocker):
    _mock_subprocess_ok(mocker)
    _write_run_manifest(tmp_path / "runs", kept=2)
    _write_fixtures(tmp_path / "fixtures", statuses=["CANC", "PST"])
    result = check_api_football_freshness(
        runs_dir=tmp_path / "runs",
        fixtures_dir=tmp_path / "fixtures",
    )
    assert result is False


def test_api_freshness_true_when_all_finished(tmp_path, mocker):
    _mock_subprocess_ok(mocker)
    _write_run_manifest(tmp_path / "runs", kept=3)
    _write_fixtures(tmp_path / "fixtures", statuses=["FT", "AET", "PEN"])
    result = check_api_football_freshness(
        runs_dir=tmp_path / "runs",
        fixtures_dir=tmp_path / "fixtures",
    )
    assert result is True


def test_api_freshness_true_with_finished_and_cancelled_mix(tmp_path, mocker):
    _mock_subprocess_ok(mocker)
    _write_run_manifest(tmp_path / "runs", kept=3)
    _write_fixtures(tmp_path / "fixtures", statuses=["FT", "CANC", "FT"])
    result = check_api_football_freshness(
        runs_dir=tmp_path / "runs",
        fixtures_dir=tmp_path / "fixtures",
    )
    assert result is True


# ---------------------------------------------------------------------------
# run_elo_ingestion
# ---------------------------------------------------------------------------

def test_run_elo_ingestion_calls_subprocess_with_overwrite_env(mocker):
    mock_run = mocker.patch(
        "subprocess.run",
        return_value=MagicMock(returncode=0, stderr="", stdout=""),
    )
    run_elo_ingestion()

    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert call_kwargs.kwargs.get("env", {}).get("ELO_OVERWRITE") == "1"
    cmd = call_kwargs.args[0]
    assert "src.ingestion.download_elo_tsvs" in cmd


def test_run_elo_ingestion_raises_on_failure(mocker):
    mocker.patch(
        "subprocess.run",
        return_value=MagicMock(returncode=1, stderr="some error", stdout=""),
    )
    with pytest.raises(RuntimeError, match="ELO ingestion failed"):
        run_elo_ingestion()


# ---------------------------------------------------------------------------
# run_dvc_pipeline
# ---------------------------------------------------------------------------

def test_run_dvc_pipeline_commits_when_changes_staged(mocker):
    call_log = []

    def fake_run(cmd, **kwargs):
        call_log.append(cmd)
        if cmd == ["git", "diff", "--cached", "--quiet"]:
            return MagicMock(returncode=1)  # staged changes exist
        return MagicMock(returncode=0)

    mocker.patch("subprocess.run", side_effect=fake_run)
    run_dvc_pipeline()

    assert ["dvc", "add", "data/raw"] in call_log
    assert ["dvc", "repro"] in call_log
    assert ["dvc", "push"] in call_log
    assert any("commit" in cmd for cmd in call_log)
    assert sum(1 for cmd in call_log if cmd[0] == "git" and "push" in cmd) >= 1


def test_run_dvc_pipeline_skips_commit_when_nothing_staged(mocker):
    call_log = []

    def fake_run(cmd, **kwargs):
        call_log.append(cmd)
        if cmd == ["git", "diff", "--cached", "--quiet"]:
            return MagicMock(returncode=0)  # nothing staged
        return MagicMock(returncode=0)

    mocker.patch("subprocess.run", side_effect=fake_run)
    run_dvc_pipeline()

    assert not any("commit" in cmd for cmd in call_log)


# ---------------------------------------------------------------------------
# main (OR gate: pipeline runs if at least one source is fresh)
# ---------------------------------------------------------------------------

def test_main_skips_when_neither_source_fresh(mocker):
    mocker.patch("src.pipeline.trigger.check_elo_freshness", return_value=False)
    mocker.patch("src.pipeline.trigger.check_api_football_freshness", return_value=False)
    mock_elo = mocker.patch("src.pipeline.trigger.run_elo_ingestion")
    mock_pipeline = mocker.patch("src.pipeline.trigger.run_dvc_pipeline")

    assert main() == 0
    mock_elo.assert_not_called()
    mock_pipeline.assert_not_called()


def test_main_runs_when_only_elo_fresh(mocker):
    mocker.patch("src.pipeline.trigger.check_elo_freshness", return_value=True)
    mocker.patch("src.pipeline.trigger.check_api_football_freshness", return_value=False)
    mock_elo = mocker.patch("src.pipeline.trigger.run_elo_ingestion")
    mock_pipeline = mocker.patch("src.pipeline.trigger.run_dvc_pipeline")
    mocker.patch("src.pipeline.trigger.dispatch_training_or_inference")

    assert main() == 0
    mock_elo.assert_called_once()
    mock_pipeline.assert_called_once()


def test_main_runs_when_only_api_fresh(mocker):
    mocker.patch("src.pipeline.trigger.check_elo_freshness", return_value=False)
    mocker.patch("src.pipeline.trigger.check_api_football_freshness", return_value=True)
    mock_elo = mocker.patch("src.pipeline.trigger.run_elo_ingestion")
    mock_pipeline = mocker.patch("src.pipeline.trigger.run_dvc_pipeline")
    mocker.patch("src.pipeline.trigger.dispatch_training_or_inference")

    assert main() == 0
    mock_elo.assert_not_called()
    mock_pipeline.assert_called_once()


def test_main_runs_pipeline_when_both_fresh(mocker):
    mocker.patch("src.pipeline.trigger.check_elo_freshness", return_value=True)
    mocker.patch("src.pipeline.trigger.check_api_football_freshness", return_value=True)
    mock_elo = mocker.patch("src.pipeline.trigger.run_elo_ingestion")
    mock_pipeline = mocker.patch("src.pipeline.trigger.run_dvc_pipeline")
    mocker.patch("src.pipeline.trigger.dispatch_training_or_inference")

    assert main() == 0
    mock_elo.assert_called_once()
    mock_pipeline.assert_called_once()


def test_main_passes_mode_to_dispatch(mocker):
    mocker.patch("src.pipeline.trigger.check_elo_freshness", return_value=True)
    mocker.patch("src.pipeline.trigger.check_api_football_freshness", return_value=True)
    mocker.patch("src.pipeline.trigger.run_elo_ingestion")
    mocker.patch("src.pipeline.trigger.run_dvc_pipeline")
    mock_dispatch = mocker.patch("src.pipeline.trigger.dispatch_training_or_inference")

    main(mode="inference_only")
    mock_dispatch.assert_called_once_with(mode="inference_only")


def test_main_rejects_invalid_mode():
    assert main(mode="bad_mode") == 1


# ---------------------------------------------------------------------------
# dispatch: inference wiring
# ---------------------------------------------------------------------------

def test_dispatch_inference_only_calls_inference(mocker):
    mocker.patch("src.models.data_split.load_gold", return_value=MagicMock(__len__=lambda s: 100))
    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mock_inference = mocker.patch("src.inference.run.run_inference_and_simulation", return_value="inf_run")
    mock_monitor = mocker.patch("src.monitoring.monitor.run_monitoring_step")
    dispatch_training_or_inference(mode="inference_only")
    assert mock_inference.call_count == 2
    mock_inference.assert_any_call(cadence_mode="frozen")
    mock_inference.assert_any_call(cadence_mode="per_round")
    mock_monitor.assert_called_once()


def test_dispatch_auto_below_threshold_calls_inference(mocker, monkeypatch):
    import mlflow
    monkeypatch.setenv("RETRAIN_THRESHOLD", "10")  # delta=5 < 10 → inference only
    mocker.patch("src.models.data_split.load_gold", return_value=MagicMock(__len__=lambda s: 105))
    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mocker.patch("src.models.mlflow_utils.get_latest_production_run_id", return_value="prod_run")
    # Pre-A.10: champion_per_round not yet assigned → legacy path
    mocker.patch("src.pipeline.trigger._last_per_round_refit_matchday", return_value=None)
    mock_client = MagicMock()
    mock_client.get_run.return_value.data.params = {"gold_row_count": "100"}
    mocker.patch("mlflow.tracking.MlflowClient", return_value=mock_client)
    mock_inference = mocker.patch("src.inference.run.run_inference_and_simulation", return_value="inf_run")
    mock_monitor = mocker.patch("src.monitoring.monitor.run_monitoring_step")
    mock_shadow = mocker.patch("src.models.pipeline.run_shadow_refit")
    dispatch_training_or_inference(mode="auto")
    assert mock_inference.call_count == 2
    mock_inference.assert_any_call(cadence_mode="frozen")
    mock_inference.assert_any_call(cadence_mode="per_round")
    mock_monitor.assert_called_once()
    mock_shadow.assert_not_called()  # below threshold → no shadow refit


def test_dispatch_auto_full_pipeline_then_inference(mocker):
    mocker.patch("src.models.data_split.load_gold", return_value=MagicMock(__len__=lambda s: 100))
    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mocker.patch("src.models.mlflow_utils.get_latest_production_run_id", return_value=None)
    mock_full = mocker.patch("src.models.pipeline.run_full_pipeline")
    mock_shadow = mocker.patch("src.models.pipeline.run_shadow_refit")
    mock_inference = mocker.patch("src.inference.run.run_inference_and_simulation", return_value="inf_run")
    mock_monitor = mocker.patch("src.monitoring.monitor.run_monitoring_step")
    dispatch_training_or_inference(mode="auto")
    mock_full.assert_called_once()
    mock_shadow.assert_called_once()
    assert mock_inference.call_count == 2
    mock_inference.assert_any_call(cadence_mode="frozen")
    mock_inference.assert_any_call(cadence_mode="per_round")
    mock_monitor.assert_called_once()


def test_dispatch_auto_refit_threshold_calls_shadow_refit(mocker):
    """Path B (pre-A.10): champion refit + shadow refit + inference + monitoring."""
    import mlflow
    mock_df = MagicMock(__len__=lambda s: 110)
    mocker.patch("src.models.data_split.load_gold", return_value=mock_df)
    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mocker.patch("src.models.mlflow_utils.get_latest_production_run_id", return_value="prod_run")
    # Pre-A.10: champion_per_round not yet assigned → legacy path
    mocker.patch("src.pipeline.trigger._last_per_round_refit_matchday", return_value=None)
    mock_client = MagicMock()
    mock_client.get_run.return_value.data.params = {"gold_row_count": "100"}
    mocker.patch("mlflow.tracking.MlflowClient", return_value=mock_client)
    mock_refit = mocker.patch("src.models.pipeline.run_champion_refit")
    mock_shadow = mocker.patch("src.models.pipeline.run_shadow_refit")
    mock_inference = mocker.patch("src.inference.run.run_inference_and_simulation", return_value="inf_run")
    mock_monitor = mocker.patch("src.monitoring.monitor.run_monitoring_step")
    dispatch_training_or_inference(mode="auto")
    mock_refit.assert_called_once()
    mock_shadow.assert_called_once()
    assert mock_inference.call_count == 2
    mock_inference.assert_any_call(cadence_mode="frozen")
    mock_inference.assert_any_call(cadence_mode="per_round")
    mock_monitor.assert_called_once()


def test_dispatch_monitoring_failure_does_not_propagate(mocker):
    """Best-effort monitoring: a raise inside ``run_monitoring_step`` is swallowed."""
    mocker.patch("src.models.data_split.load_gold", return_value=MagicMock(__len__=lambda s: 100))
    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mocker.patch("src.inference.run.run_inference_and_simulation", return_value="inf_run")
    mocker.patch(
        "src.monitoring.monitor.run_monitoring_step",
        side_effect=RuntimeError("boom"),
    )
    # Should not raise.
    dispatch_training_or_inference(mode="inference_only")


def test_dispatch_shadow_refit_failure_does_not_propagate(mocker):
    """Best-effort shadow refit: a raise inside ``run_shadow_refit`` is swallowed."""
    mocker.patch("src.models.data_split.load_gold", return_value=MagicMock(__len__=lambda s: 100))
    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mocker.patch("src.models.mlflow_utils.get_latest_production_run_id", return_value=None)
    mocker.patch("src.models.pipeline.run_full_pipeline")
    mocker.patch(
        "src.models.pipeline.run_shadow_refit",
        side_effect=RuntimeError("shadow boom"),
    )
    mocker.patch("src.inference.run.run_inference_and_simulation", return_value="inf_run")
    mocker.patch("src.monitoring.monitor.run_monitoring_step")
    # Should not raise.
    dispatch_training_or_inference(mode="auto")


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

def test_parse_args_defaults_to_auto():
    args = _parse_args([])
    assert args.mode == "auto"


def test_parse_args_accepts_inference_only():
    args = _parse_args(["--mode", "inference_only"])
    assert args.mode == "inference_only"


def test_parse_args_rejects_invalid_mode():
    with pytest.raises(SystemExit):
        _parse_args(["--mode", "bad"])


# ---------------------------------------------------------------------------
# B.3 — _last_per_round_refit_matchday
# ---------------------------------------------------------------------------

def test_last_per_round_matchday_returns_none_when_alias_absent(mocker):
    """pre-A.10: MlflowException on alias lookup → None."""
    import mlflow

    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.side_effect = (
        mlflow.exceptions.MlflowException("alias not found")
    )
    mocker.patch("mlflow.tracking.MlflowClient", return_value=mock_client)

    result = _last_per_round_refit_matchday()
    assert result is None


def test_last_per_round_matchday_returns_tag_when_alias_present(mocker):
    """post-A.10: alias exists with tag → returns matchday label."""
    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mock_mv = MagicMock(run_id="run-per-round")
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = mock_mv
    mock_client.get_run.return_value.data.tags = {"per_round_refit_matchday": "R32"}
    mocker.patch("mlflow.tracking.MlflowClient", return_value=mock_client)

    result = _last_per_round_refit_matchday()
    assert result == "R32"


def test_last_per_round_matchday_returns_zero_when_tag_absent(mocker):
    """Alias exists but no tag → returns '0' sentinel."""
    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mock_mv = MagicMock(run_id="run-no-tag")
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = mock_mv
    mock_client.get_run.return_value.data.tags = {}
    mocker.patch("mlflow.tracking.MlflowClient", return_value=mock_client)

    result = _last_per_round_refit_matchday()
    assert result == "0"


# ---------------------------------------------------------------------------
# B.3 — dispatch: WC boundary-gated path
# ---------------------------------------------------------------------------

def test_dispatch_wc_boundary_fires_per_round_refit(mocker):
    """WC mode: matchday advanced → per-round refit fires, frozen never refits."""
    mocker.patch("src.models.data_split.load_gold", return_value=MagicMock(__len__=lambda s: 200))
    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mocker.patch("src.models.mlflow_utils.get_latest_production_run_id", return_value="prod_run")
    # post-A.10: last refit was matchday "1", now matchday "2"
    mocker.patch(
        "src.pipeline.trigger._last_per_round_refit_matchday", return_value="1",
    )
    mocker.patch(
        "src.inference.features.parse_wc_results",
        return_value={"next_matchday": 2, "group_results": {}, "ko_results": {}, "finished_fixtures": []},
    )
    mock_per_round = mocker.patch("src.models.pipeline.run_per_round_refit")
    mock_champ_refit = mocker.patch("src.models.pipeline.run_champion_refit")
    mock_inference = mocker.patch("src.inference.run.run_inference_and_simulation", return_value="inf")
    mock_monitor = mocker.patch("src.monitoring.monitor.run_monitoring_step")

    dispatch_training_or_inference(mode="auto")

    mock_per_round.assert_called_once_with(mocker.ANY, matchday="2")
    mock_champ_refit.assert_not_called()
    assert mock_inference.call_count == 2
    mock_monitor.assert_called_once()


def test_dispatch_wc_no_boundary_skips_refit(mocker):
    """WC mode: matchday unchanged → no refit, inference only."""
    mocker.patch("src.models.data_split.load_gold", return_value=MagicMock(__len__=lambda s: 200))
    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mocker.patch("src.models.mlflow_utils.get_latest_production_run_id", return_value="prod_run")
    mocker.patch(
        "src.pipeline.trigger._last_per_round_refit_matchday", return_value="2",
    )
    mocker.patch(
        "src.inference.features.parse_wc_results",
        return_value={"next_matchday": 2, "group_results": {}, "ko_results": {}, "finished_fixtures": []},
    )
    mock_per_round = mocker.patch("src.models.pipeline.run_per_round_refit")
    mock_inference = mocker.patch("src.inference.run.run_inference_and_simulation", return_value="inf")
    mock_monitor = mocker.patch("src.monitoring.monitor.run_monitoring_step")

    dispatch_training_or_inference(mode="auto")

    mock_per_round.assert_not_called()
    assert mock_inference.call_count == 2
    mock_monitor.assert_called_once()


def test_dispatch_wc_complete_skips_refit(mocker):
    """WC mode: tournament Complete → no refit even though matchday changed."""
    mocker.patch("src.models.data_split.load_gold", return_value=MagicMock(__len__=lambda s: 200))
    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mocker.patch("src.models.mlflow_utils.get_latest_production_run_id", return_value="prod_run")
    mocker.patch(
        "src.pipeline.trigger._last_per_round_refit_matchday", return_value="SF",
    )
    mocker.patch(
        "src.inference.features.parse_wc_results",
        return_value={"next_matchday": "Complete", "group_results": {}, "ko_results": {}, "finished_fixtures": []},
    )
    mock_per_round = mocker.patch("src.models.pipeline.run_per_round_refit")
    mock_inference = mocker.patch("src.inference.run.run_inference_and_simulation", return_value="inf")
    mock_monitor = mocker.patch("src.monitoring.monitor.run_monitoring_step")

    dispatch_training_or_inference(mode="auto")

    mock_per_round.assert_not_called()
    assert mock_inference.call_count == 2


def test_dispatch_per_round_refit_failure_does_not_propagate(mocker):
    """Best-effort: a raise inside run_per_round_refit is swallowed."""
    mocker.patch("src.models.data_split.load_gold", return_value=MagicMock(__len__=lambda s: 200))
    mocker.patch("src.models.mlflow_utils.setup_mlflow")
    mocker.patch("src.models.mlflow_utils.get_latest_production_run_id", return_value="prod_run")
    mocker.patch(
        "src.pipeline.trigger._last_per_round_refit_matchday", return_value="1",
    )
    mocker.patch(
        "src.inference.features.parse_wc_results",
        return_value={"next_matchday": "R32", "group_results": {}, "ko_results": {}, "finished_fixtures": []},
    )
    mocker.patch(
        "src.models.pipeline.run_per_round_refit",
        side_effect=RuntimeError("refit exploded"),
    )
    mock_inference = mocker.patch("src.inference.run.run_inference_and_simulation", return_value="inf")
    mocker.patch("src.monitoring.monitor.run_monitoring_step")

    # Must not raise; inference continues
    dispatch_training_or_inference(mode="auto")
    assert mock_inference.call_count == 2


# ---------------------------------------------------------------------------
# Concurrency lock
# ---------------------------------------------------------------------------

def test_main_skips_when_lock_held(mocker):
    """If _try_acquire_lock returns None the tick is skipped (exit 0)."""
    mocker.patch("src.pipeline.trigger._try_acquire_lock", return_value=None)
    mock_elo = mocker.patch("src.pipeline.trigger.check_elo_freshness")
    mock_api = mocker.patch("src.pipeline.trigger.check_api_football_freshness")

    result = main()

    assert result == 0
    mock_elo.assert_not_called()
    mock_api.assert_not_called()


def test_try_acquire_lock_returns_handle_when_free(tmp_path, mocker):
    """_try_acquire_lock returns a file handle when no lock is held."""
    mocker.patch("src.pipeline.trigger._LOCK_FILE", tmp_path / ".trigger.lock")
    fh = _try_acquire_lock()
    assert fh is not None
    fh.close()


def test_try_acquire_lock_returns_none_when_held(tmp_path, mocker):
    """_try_acquire_lock returns None when another handle holds the lock."""
    mocker.patch("src.pipeline.trigger._LOCK_FILE", tmp_path / ".trigger.lock")
    first = _try_acquire_lock()
    assert first is not None
    try:
        second = _try_acquire_lock()
        assert second is None
    finally:
        first.close()
