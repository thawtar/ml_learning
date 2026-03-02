# DVC (Data Version Control) Guide

## What is DVC?

DVC is a version control system for data and ML pipelines. It works alongside Git:
- **Git** tracks code (small files)
- **DVC** tracks data and models (large files)

---

## Step 1: Track Raw Data

```bash
dvc add data/customer_churn_dataset-training-master.csv
```

### What happens:
1. DVC calculates a hash (MD5) of the file
2. Creates `data/customer_churn_dataset-training-master.csv.dvc` — a small pointer file containing the hash
3. Adds the actual CSV to `.gitignore` (so git ignores the large file)
4. Moves the actual data to `.dvc/cache/`

```
Before:                          After:
data/                            data/
└── customer_churn...csv         ├── customer_churn...csv.dvc  ← pointer (tiny)
                                 └── .gitignore

                                 .dvc/cache/
                                 └── ab/cd1234...  ← actual data (cached)
```

### Why:
Git tracks the small `.dvc` file. The actual large data is cached separately.

---

## Step 2: Run Pipeline

```bash
dvc repro
```

### What happens:
1. DVC reads `dvc.yaml`
2. Checks each stage's dependencies (`deps`)
3. Runs stages in order if inputs changed

```
Checks dvc.yaml:
┌─────────────────────────────────────────┐
│ Stage: preprocess                       │
│   deps: data/customer_churn...csv  ✓    │ ← changed?
│         src/preprocess.py          ✓    │ ← changed?
│   → Runs: python src/preprocess.py      │
│   → Outputs: data_processed/*           │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ Stage: train                            │
│   deps: data_processed/*           ✓    │ ← changed?
│         src/train.py               ✓    │ ← changed?
│   params: model.C, model.penalty   ✓    │ ← changed?
│   → Runs: python src/train.py           │
│   → Outputs: models/model.pkl           │
└─────────────────────────────────────────┘
```

### Key feature:
If you run `dvc repro` again without changing anything, DVC skips everything (cached).

---

## Step 3: Check Status

```bash
dvc status
```

### What happens:
- Compares current files against cached hashes
- Shows which stages are out of sync

### Example output:
```
preprocess:
    changed deps:
        modified:  src/preprocess.py   ← you edited this
```

---

## Step 4: Change Hyperparameters & Re-run

```bash
# Edit params.yaml: change C from 1.0 to 0.5
dvc repro
```

### What happens:
1. DVC detects `params.yaml` changed
2. Skips `preprocess` stage (data unchanged)
3. Only runs `train` stage (params changed)

```
preprocess: skipped (cached)  ← fast!
train: running...             ← only this runs
```

---

## Step 5: View Pipeline DAG

```bash
dvc dag
```

### Output:
```
+--------------------+
| data/customer...csv|
+--------------------+
          |
          v
   +------------+
   | preprocess |
   +------------+
          |
          v
   +------------+
   |   train    |
   +------------+
```

---

## Step 6: Remote Storage (Optional)

```bash
# Configure remote (S3, GCS, or local folder)
dvc remote add -d myremote s3://my-bucket/dvc

# Push data to remote
dvc push

# Pull data on another machine
dvc pull
```

### What happens:
- `dvc push` — uploads cached data to remote storage
- `dvc pull` — downloads data from remote using `.dvc` pointer files

```
Your Machine                    Remote (S3/GCS)
.dvc/cache/                     s3://bucket/dvc/
└── ab/cd1234...   ──push──→    └── ab/cd1234...
                   ←─pull──
```

### Why:
Team members clone the git repo, run `dvc pull`, and get the exact same data.

---

## Summary Flow

```
┌─────────────────────────────────────────────────────────┐
│  1. dvc add data.csv         → track large files        │
│  2. git add/commit *.dvc     → version pointers         │
│  3. dvc repro                → run pipeline             │
│  4. dvc push                 → backup data to remote    │
│  5. (teammate) git pull      → get .dvc files           │
│  6. (teammate) dvc pull      → get actual data          │
│  7. (teammate) dvc repro     → reproduce results        │
└─────────────────────────────────────────────────────────┘
```

---

## Common Commands Reference

| Command | Description |
|---------|-------------|
| `dvc init` | Initialize DVC in a project |
| `dvc add <file>` | Track a file with DVC |
| `dvc repro` | Run the pipeline |
| `dvc status` | Check what's changed |
| `dvc dag` | View pipeline graph |
| `dvc push` | Upload data to remote |
| `dvc pull` | Download data from remote |
| `dvc remote add` | Configure remote storage |
| `dvc metrics show` | Show metrics |
| `dvc params diff` | Compare parameters between runs |

---

## Your Project's dvc.yaml Explained

```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py      # command to run
    deps:                               # re-run if these change
      - data/customer_churn_data.csv
      - src/preprocess.py
    outs:                               # files this stage produces
      - data_processed/X_train.csv
      - data_processed/X_test.csv
      - data_processed/y_train.csv
      - data_processed/y_test.csv
      - data_processed/encoders.pkl
      - data_processed/scaler.pkl

  train:
    cmd: python src/train.py
    deps:
      - data_processed/X_train.csv
      - data_processed/X_test.csv
      - data_processed/y_train.csv
      - data_processed/y_test.csv
      - src/train.py
    params:                             # re-run if these params change
      - model.C
      - model.penalty
    outs:
      - models/model.pkl
```

---

## DVC vs Git

| Aspect | Git | DVC |
|--------|-----|-----|
| Tracks | Code, configs | Data, models |
| File size | Small (<100MB) | Large (GB+) |
| Storage | GitHub/GitLab | S3, GCS, Azure, local |
| Versioning | Full file history | Hash-based pointers |
