#!/usr/bin/env sh
# Container entrypoint.
# Production: pulls the latest versioned data from the DVC remote (GCS) first.
# Local dev:  set SKIP_DVC_PULL=1 and mount -v $(pwd)/data:/app/data instead.
set -e

# Configure DVC remote auth from env vars (local: .env, cloud: Secret Manager).
# DVC has no credential env vars of its own, so write them to the git-ignored
# .dvc/config.local at runtime. Without this, dvc push/pull is unauthenticated.
if [ -n "${DAGSHUB_TOKEN}" ]; then
    echo "Configuring DVC remote auth..."
    dvc remote modify --local origin auth basic
    dvc remote modify --local origin user "${DAGSHUB_USERNAME}"
    dvc remote modify --local origin password "${DAGSHUB_TOKEN}"

    # The image bakes the code but excludes .git (it has its own throwaway repo
    # from `git init`). Reconnect to the real remote history so run_dvc_pipeline's
    # git commit/push of the data pointers fast-forwards onto `thesis` instead of
    # being rejected as an unrelated root commit.
    echo "Reconnecting git to remote history (thesis)..."
    GIT_REPO="https://${DAGSHUB_USERNAME}:${DAGSHUB_TOKEN}@dagshub.com/tomppa999/MLOps_FootballWCPredictor.git"
    git remote add origin "${GIT_REPO}" 2>/dev/null || git remote set-url origin "${GIT_REPO}"
    git fetch --depth 1 origin thesis
    git reset --mixed FETCH_HEAD
    git branch -M thesis
    git branch --set-upstream-to=origin/thesis thesis
fi

if [ "${SKIP_DVC_PULL}" != "1" ]; then
    echo "Pulling data from DVC remote..."
    dvc pull
fi

exec python -m src.pipeline.trigger --mode="${PIPELINE_MODE:-auto}"
