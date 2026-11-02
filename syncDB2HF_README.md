# syncDB2HF: Documentation

`syncDB2HF.py` is the central maintenance script for the LEMUR NN Dataset. It ingests new training statistics into the database, verifies that all local artifacts are safely backed up, generates an audit report, compresses and uploads the updated database to Hugging Face, and finally deletes local files only when every safety condition is satisfied.

> **Safety guarantee:** Local files are never deleted unless all 5 conditions pass simultaneously: the artifact is marked SAFE, it is verified in the database, the database was compressed successfully, the upload to Hugging Face succeeded, and the Excel audit was generated. If any one condition fails, nothing is deleted.

> **Authorization:** The script publishes to Hugging Face, so it runs only with a valid Hugging Face token. Without one it stops immediately, before any phase runs, and reports that the user is not authorized to synchronize and publish.

The earlier publishing command, `python -m ab.nn.util.hf.DB2HF --remove_gen_folders --HF_TOKEN <token>`, remains available.

---

## Requirements

- Python 3.10 or higher
- The `nn-dataset` repository cloned locally
- A Python virtual environment with the project dependencies installed, activated
- A valid Hugging Face token
- Dependencies: `pandas`, `openpyxl`, `zstandard`, `huggingface_hub`

---

## Setup

Navigate to the repository folder and activate your virtual environment:

```bash
cd path/to/nn-dataset
source venv/bin/activate
```

Provide your Hugging Face token through an environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

Alternatively, pass it on the command line:

```bash
python syncDB2HF.py --HF_TOKEN hf_your_token_here
```

---

## How to Run

**Full pipeline (all 10 phases, including upload and deletion):**

```bash
python syncDB2HF.py
```

At the start, the token is checked with Hugging Face. A missing or invalid token ends the run with exit code 1.

---

## The 10 Phases

The pipeline runs 10 phases in sequence. Deletion in Phase 10 is gated on the outcome of the earlier phases.

---

### Phase 1: Inventory Scan

Scans three local directories and builds an in-memory list of everything present:

- `ab/nn/nn/`: all model `.py` files
- `ab/nn/stat/train/`: all statistic folders (named `<task>_<dataset>_<metric>_<ModelName>`)
- `ab/nn/transform/`: all transform `.py` files

**Output:** counts of models, stat folders, and transforms found locally.

---

### Phase 2: HF Fallback (Ensure DB Exists)

Checks whether the local database `db/ab.nn.db` exists.

- If it **already exists**, this phase is skipped.
- If it **does not exist**, the database is downloaded from the Hugging Face LEMUR\_DB repository. The file name is `ab.nn.zst-<version>`, for example `ab.nn.zst-2.2.9`.

The version is determined as follows:

1. If `nn-dataset` is installed as a dependency (`pip install nn-dataset`), the installed package version is used.
2. Otherwise, the project is used as a standalone clone and the version is read from the `version` file in the repository root.

If that exact version has not been published yet, for example for a new release whose database has not been uploaded, the newest published version below it is downloaded and used as the baseline. A warning is logged in this case. Phase 9 then uploads the result under the current version name.

This ensures the pipeline always has a database to work with, even on a fresh clone.

---

### Phase 3: Stat Ingestion

Transfers any new local training statistics into the database before verification runs.

Uses the project's own `json_train_to_db()` function, which reads every JSON file from `ab/nn/stat/train/` and inserts the results into the `stat` and `train_stat` tables. Existing rows are not duplicated.

**Why this runs before verification:** Phase 6 checks whether local stats exist in the database. Running ingestion first ensures that newly added stats are already in the database when that check happens, so they are correctly marked as verified.

---

### Phase 4: Dependency Mapping

Links models, stat folders, and transforms together by reading every JSON file inside each stat folder and extracting the `"transform"` field.

After this phase the pipeline knows:
- Which stat folders belong to which model
- Which transforms are referenced by which stat folders

---

### Phase 5: SAFE / KEEP Logic

Decides which artifacts are candidates for archival based on local completeness.

**A model is marked SAFE if:**
- Its `.py` file exists locally
- It has at least one stat folder locally
- Every one of its stat folders has at least one JSON file

**A transform is marked SAFE if:**
- Its `.py` file exists locally
- Every stat folder that references it belongs to a SAFE model

Artifacts marked **KEEP** are never touched, regardless of what happens in later phases.

---

### Phase 6: DB Verification

Cross-checks every SAFE artifact against the central database to confirm it has been archived. The verification method depends on the artifact type:

| Artifact | Method | Reason |
|---|---|---|
| Base model (e.g. `AlexNet.py`) | SHA256 hash of local `.py` vs `nn_code` stored in the database | Source code is stored in the database for base models |
| Generated variant (e.g. `ast-dimension-AlexNet-9244124f15c45d6add6bcb95d922a9c7`) | Direct SQL `SELECT 1 FROM stat WHERE nn = ?` | Variants have no source code stored in the database, only their training stats |
| Stat folder | `(task, dataset, nn)` tuple lookup in the `stat` table | Confirms training data was archived |
| Transform | SHA256 hash of local `.py` vs `transform_code` stored in the database | Source code is stored in the database for transforms |

Only artifacts that pass both Phase 5 (SAFE) **and** Phase 6 (DB Verified) are eligible for deletion in Phase 10.

---

### Phase 7: Excel Audit Report

Generates a timestamped Excel file (`LEMUR_Deep_Audit_<timestamp>.xlsx`) with three sheets:

| Sheet | Contents |
|---|---|
| **Models & Stats** | Every model, whether it exists locally, how many stat folders it has, and its SAFE/KEEP decision |
| **Transform Deep Dive** | Every JSON file with its model, stat folder, transform name, and action |
| **DB Verification Audit** | Every artifact with its verification method, DB verification result, and final DELETE/KEEP decision |

This file is saved **locally only** and is never uploaded to Hugging Face.

---

### Phase 8: Compress Database

Compresses `db/ab.nn.db` into a versioned file (e.g. `db/ab.nn.zst-2.2.9`) using the project's own `compress()` helper (`zstandard`, level 16, all CPU cores).

The version in the file name is determined as described in Phase 2: the installed `nn-dataset` package version if there is one, otherwise the `version` file. The original database is never deleted at this step.

---

### Phase 9: Upload to Hugging Face

Uploads the compressed file from Phase 8 to the LEMUR\_DB Hugging Face repository using the project's own `upload_file()` helper.

- Uses the token that was validated at the start of the run
- Skipped automatically if Phase 7 (audit) or Phase 8 (compression) failed
- Skipped if the version in use (from an installed `nn-dataset` package) differs from the `version` file of the repository. The file is named after the version, so uploading would replace the database of that other version. Uninstall the package (`pip uninstall nn-dataset`) and run again
- The Excel audit report is **not** uploaded; it stays local only

---

### Phase 10: Local Deletion

Deletes local files only if **all 5 conditions** are satisfied simultaneously:

| Condition | What is checked |
|---|---|
| 1. SAFE | Phase 5 marked this artifact as SAFE TO PROCESS |
| 2. DB Verified | Phase 6 confirmed this artifact exists in the database |
| 3. Compression OK | Phase 8 completed successfully |
| 4. Upload OK | Phase 9 completed successfully |
| 5. Audit OK | Phase 7 generated the Excel report successfully |

If any one of these conditions is not met, **nothing is deleted** and a warning is logged.

**What gets deleted when all conditions pass:**
- The model `.py` file from `ab/nn/nn/`
- The JSON files inside each verified stat folder, and the folder itself if it becomes empty
- The transform `.py` file from `ab/nn/transform/`

---

## Output Files

| File | Created by | Location | Description |
|---|---|---|---|
| `LEMUR_Deep_Audit_<timestamp>.xlsx` | Phase 7 | Repo root | Full audit with 3 sheets |
| `db/ab.nn.zst-<version>` | Phase 8 | `db/` folder | Compressed database for upload |

---

## Pipeline Flow Summary

```
Token check   valid Hugging Face token required
    ↓
Phase 1   Inventory Scan
    ↓
Phase 2   HF Fallback: download DB if missing
    ↓
Phase 3   Stat Ingestion: add new local stats to the DB
    ↓
Phase 4   Dependency Mapping
    ↓
Phase 5   SAFE / KEEP Logic
    ↓
Phase 6   DB Verification
    ↓
Phase 7   Excel Audit Report
    ↓
Phase 8   Compress DB
    ↓
Phase 9   Upload to Hugging Face
    ↓
Phase 10  Local Deletion (only if all 5 safety conditions pass)
```

---

## Security Notes

- The Hugging Face token is never hardcoded. It must be supplied via the `HF_TOKEN` environment variable or the `--HF_TOKEN` command-line argument.
- The script does not run without a valid token.
- No files are deleted unless the upload to Hugging Face has been confirmed successful in the same run.
- The audit Excel file is always generated before any deletion attempt, providing a permanent record of every decision made.
