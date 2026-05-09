# NREL original reference-data restore and command notes

This note captures the provenance-safe, separate-directory workflow for the original NREL example TFRecords.

## Scope
- Keep restored reference TFRecords under `reference_nrel_example_data/`.
- Keep paired inference outputs under `data_out_nrel_original/`.
- Keep scalar-trained models under `models_nrel_original/`.
- Keep comparison/analysis outputs under `analysis_nrel_original/`.
- Do **not** overwrite `example_data/` or any existing outputs.

## Canonical sample list status
No canonical upstream sample list confirmed.

Observed repo references only:
- `quick_figs.py` renders sample `0`.
- `scripts/run_full_experiment.sh` highlights `2 165 166` in a tie-break example.

Recommended fallback IDs until the restored upstream record count is verified:
- `0 2` because they are explicitly referenced in repo tooling and remain low-risk on smaller datasets.
- Use `165 166` only after confirming the restored TFRecord has at least `167` records.

## Corrected record-count command
Use `tf.compat.v1.io.tf_record_iterator` rather than the deprecated/internal `tf.python_io.tf_record_iterator`.

Copy-paste command:

```bash
python - <<'PY'
from pathlib import Path

files = [
    Path('reference_nrel_example_data/wind_MR-HR.tfrecord'),
    Path('reference_nrel_example_data/wind_LR-MR.tfrecord'),
    Path('reference_nrel_example_data/wind_speed_MR-HR.tfrecord'),
]

for path in files:
    exists = path.exists()
    size = path.stat().st_size if exists else 'MISSING'
    count = 'MISSING'
    if exists:
        try:
            import tensorflow as tf
            count = sum(1 for _ in tf.compat.v1.io.tf_record_iterator(str(path)))
        except Exception as exc:
            count = f'ERROR: {exc}'
    print(f'{path}\texists={exists}\tbytes={size}\trecords={count}')
PY
```

## Manual restore instructions to run outside Codex
Upstream repo path referenced by this repository's README:
- GitHub repository: `NREL/PhIRE`
- Expected upstream directory path inside that repo: `example_data/`

Because upstream GitHub access could not be verified from this environment, the raw file URLs below are the standard GitHub raw paths inferred from the repository name and file layout used throughout this repo; verify them in your browser if needed before running large downloads.

Likely raw URLs:
- `https://raw.githubusercontent.com/NREL/PhIRE/master/example_data/wind_MR-HR.tfrecord`
- `https://raw.githubusercontent.com/NREL/PhIRE/master/example_data/wind_LR-MR.tfrecord`

If the upstream default branch is `main` instead of `master`, substitute `/main/` for `/master/` in the URLs.

### Option A: `wget`

```bash
mkdir -p reference_nrel_example_data && \
wget -O reference_nrel_example_data/wind_MR-HR.tfrecord \
  https://raw.githubusercontent.com/NREL/PhIRE/master/example_data/wind_MR-HR.tfrecord && \
wget -O reference_nrel_example_data/wind_LR-MR.tfrecord \
  https://raw.githubusercontent.com/NREL/PhIRE/master/example_data/wind_LR-MR.tfrecord
```

### Option B: `curl`

```bash
mkdir -p reference_nrel_example_data && \
curl -fL https://raw.githubusercontent.com/NREL/PhIRE/master/example_data/wind_MR-HR.tfrecord \
  -o reference_nrel_example_data/wind_MR-HR.tfrecord && \
curl -fL https://raw.githubusercontent.com/NREL/PhIRE/master/example_data/wind_LR-MR.tfrecord \
  -o reference_nrel_example_data/wind_LR-MR.tfrecord
```

### Option C: shallow `git clone` then copy only the needed files

```bash
rm -rf /tmp/PhIRE_upstream && \
git clone --depth 1 https://github.com/NREL/PhIRE.git /tmp/PhIRE_upstream && \
mkdir -p reference_nrel_example_data && \
cp /tmp/PhIRE_upstream/example_data/wind_MR-HR.tfrecord reference_nrel_example_data/ && \
cp /tmp/PhIRE_upstream/example_data/wind_LR-MR.tfrecord reference_nrel_example_data/
```

Optional scalar-speed TFRecord creation after restore:

```bash
python scripts/build_wind_speed_tfrecord.py \
  --input reference_nrel_example_data/wind_MR-HR.tfrecord \
  --output reference_nrel_example_data/wind_speed_MR-HR.tfrecord
```

## Conditional compare commands
First verify the restored record count with the command above.

### If the restored upstream set has at least 167 records

```bash
python scripts/compare_selected_wind_samples.py \
  --cnn-dir data_out_nrel_original/wind_mrhr_cnn \
  --gan-dir data_out_nrel_original/wind_mrhr_gan \
  --scalar-dir data_out_nrel_original/wind_speed_mrhr_cnn \
  --samples 0 2 165 166 \
  --outdir analysis_nrel_original/selected_samples
```

### Fallback if the restored upstream set is smaller than 167 records

```bash
python scripts/compare_selected_wind_samples.py \
  --cnn-dir data_out_nrel_original/wind_mrhr_cnn \
  --gan-dir data_out_nrel_original/wind_mrhr_gan \
  --scalar-dir data_out_nrel_original/wind_speed_mrhr_cnn \
  --samples 0 2 \
  --outdir analysis_nrel_original/selected_samples_small
```
