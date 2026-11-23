# Thesis input archive

Two local Git tags identify the 23 November 2026 checkpoint. `thesis-before-joint-poisson` preserves the working analysis at commit `c8ca67b5`, including the 90-minute analysis and the Git-managed tables and figures. `thesis-joint-inputs-v1` adds the three DVC pointers below. Both tags share the same historical analysis code and lock file.

- **`baseline.dvc`** stores byte copies from the restored baseline. Paths under `baseline/` mirror the repository. They include the pre-rescoring match table, full entropy trajectory, official paired cycles and state flags, regulation-time data, and reference tables. `baseline/source.json` records the Git revision and SHA-256 of each archived payload.
- **`poisson.dvc`** stores nine numeric model states and their registry manifest, feature order and runtime versions, 413 historical Gold snapshots, the original Gold commit history, the trial’s logged cycle map, Strand 5’s stored marginal matrix and archived Poisson advancement files from Strands 2 and 5. Select the applicable strand for each run. Some frozen cycles have reference files in both. Keep the archived synthetic per-round predictions as evidence of the reported cache issue.
- **`sensitivity.tar.gz.dvc`** stores the three-seed trial scripts, DVC configuration and lock, tests, model inputs, prepared state arrays, 1,956 simulation records, tables and figures. To reconstruct the trial’s original input layout, take the Gold files listed in its `inputs/gold_index.json` from the Poisson archive. Its scripts and provenance records refer to the original repository path, DVC cache and historical remote. The preserved outputs can be read directly from the tarball.

The existing `data/analysis/live.dvc`, `data/reconstruction/inputs.dvc` and `data/raw.dvc` supply the live exports, logged all-model marginal matrix, settlement data, outage regime manifest and raw fixtures. The baseline Git tag supplies the tournament configuration, team mapping, reconstruction rules and the original feature code.

## Gold and model provenance

Each `poisson/gold/<DVC directory hash>/matches.parquet` has its original content MD5. The 412 snapshot identities in `poisson/gold_index_trial.json` cover the tournament cycle times. Frozen match reconstruction at kickoff also requires `ccc11befb95c86553ff61d8da41f446a.dir`, introduced by commit `7eab760bf43bb00a93b2b5749bf06ba011d55f77`. Its original directory record is next to the parquet (MD5 `78cc4c811e0e6a82deec6bc0d394abd0`). `poisson/gold_history.tar.gz` contains the original commit timestamps and 454 corresponding lock files from `wc2026-end-of-tournament`. Select the last commit strictly before the relevant cutoff using the Strand 1 or Strand 5 rule.

The model manifest records source pickle SHA-256, run ID, creation time and version. For each of the nine versions, the archived numeric JSON was compared with the original scaler and coefficient bytes using the baseline’s static extractor. The archived live cycle map records the trial’s version attribution for the 845 logged Poisson cycles. The eight synthetic rows take their versions from the regime manifest. The corrected probabilities and synthetic predictions are work for the next milestone.

## Local restore

`local_store/` in the original workspace is a local DVC backup. It is independent of the working DVC cache. Use it as an explicit remote in a **separate checkout** of this repository. Replace the example path with the archive’s location.

```sh
git switch --detach thesis-joint-inputs-v1
dvc remote add --local thesis-archive /absolute/path/to/data/thesis_inputs/local_store
dvc pull -r thesis-archive
```

Use `thesis-before-joint-poisson` for a checkout of the baseline alone. The same store serves its DVC objects. On a new machine, transfer the Git tags and the store, or publish their objects to the configured project remote first. The initial checkpoint is local.
