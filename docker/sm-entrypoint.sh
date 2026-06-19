#!/usr/bin/env bash
#
# Amazon SageMaker entrypoint for the `sagemaker` Docker stage.
#
# SageMaker Training Jobs start a bring-your-own container as
# `docker run <image> train`. This script intercepts that `train` invocation and
# maps SageMaker's filesystem conventions onto the project's Hydra entrypoint
# (experiments/train.py); any other invocation is passed straight through to
# `python`, so the image still behaves like the base `runtime` stage for ad-hoc
# `docker run <image> experiments/...` use.
#
# SageMaker conventions handled here:
#   /opt/ml/input/config/hyperparameters.json  ->  Hydra `key=value` overrides
#   /opt/ml/model         final artefacts, uploaded to S3 as model.tar.gz
#   /opt/ml/checkpoints   continuously synced to checkpoint_s3_uri (spot-safe);
#                         used as results_dir so periodic checkpoints survive
#                         interruption, then resumed from on restart.
set -euo pipefail

# Anything that isn't the SageMaker `train` command is a passthrough to python,
# preserving the base image's `ENTRYPOINT ["python"]` ergonomics.
if [[ "${1:-}" != "train" ]]; then
    exec python "$@"
fi

CONFIG_DIR="/opt/ml/input/config"
HP_FILE="${CONFIG_DIR}/hyperparameters.json"
MODEL_DIR="/opt/ml/model"
CKPT_DIR="/opt/ml/checkpoints"

mkdir -p "${MODEL_DIR}" "${CKPT_DIR}"

# Translate the flat {string: string} hyperparameters object into Hydra CLI
# overrides. SageMaker-internal keys (sagemaker_*, leading underscore) are
# skipped. Emitted NUL-separated so values with spaces stay intact.
declare -a OVERRIDES=()
if [[ -f "${HP_FILE}" ]]; then
    while IFS= read -r -d '' override; do
        OVERRIDES+=("${override}")
    done < <(python - "${HP_FILE}" <<'PY'
import json
import sys

with open(sys.argv[1]) as fh:
    params = json.load(fh)

for key, value in params.items():
    if key.startswith("sagemaker_") or key.startswith("_"):
        continue
    sys.stdout.write(f"{key}={value}\x00")
PY
    )
fi

# Resume from the latest periodic checkpoint synced into /opt/ml/checkpoints
# (present after a spot interruption when checkpoint_s3_uri is reused). The
# CheckpointCallback names files `<run_name>_<steps>_steps.zip`; pick the
# highest step count.
RESUME_FROM="$(python - "${CKPT_DIR}" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
best = (None, -1)
for path in ckpt_dir.glob("*_steps.zip"):
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if match and int(match.group(1)) > best[1]:
        best = (path, int(match.group(1)))
if best[0] is not None:
    print(best[0])
PY
)"

# Outputs land in /opt/ml/checkpoints so periodic checkpoints, best_model.zip and
# the final model are continuously synced to S3 (durable across spot restarts).
OVERRIDES+=("results_dir=${CKPT_DIR}" "hydra.run.dir=${CKPT_DIR}")
if [[ -n "${RESUME_FROM}" ]]; then
    echo "sm-entrypoint: resuming from ${RESUME_FROM}"
    OVERRIDES+=("resume_from=${RESUME_FROM}")
fi

echo "sm-entrypoint: python experiments/train.py ${OVERRIDES[*]}"
python experiments/train.py "${OVERRIDES[@]}"

# Copy the final artefacts into /opt/ml/model so SageMaker produces the standard
# model.tar.gz in the job's output_path (separate from the checkpoint channel).
for artefact in best_model.zip model.zip; do
    if [[ -f "${CKPT_DIR}/${artefact}" ]]; then
        cp "${CKPT_DIR}/${artefact}" "${MODEL_DIR}/${artefact}"
    fi
done
