# Model Selection System - Execution Guide

This document outlines the step-by-step instructions to run the Similarity-Based Model Selection System.

## Prerequisites

- Python 3.x installed.
- No external dependencies required (uses standard library `sqlite3`, `json`, etc.).
- The `datasets` folder should be present in the parent directory or configured path.

## 1. Setup / Navigation

First, navigate to the directory containing the updated source code.

```powershell
cd nn-dataset\similarity-curriculum-selector
```

## 2. Execution Steps

Run the following scripts in order to build the system and generate results.

### Step 1: Ingest Data

**Description:** Scans the dataset directories, extracts model metadata (accuracy, architecture, etc.), and populates the `models` table in the SQLite database (`model_selection.db`).

```powershell
python ingest_data.py
```

### Step 2: Compute Similarity Index

**Description:** Reads the model metadata from the database, creates MinHash signatures based on architecture and configuration, and populates the `model_similarity` table with similarity edges (metric: `meta_jaccard`).

```powershell
python compute_similarity.py
```

### Step 3: Generate Curriculum

**Description:** Uses the database to generate an "Easy-to-Hard" curriculum. It picks a starting model (low accuracy) and traverses the similarity graph to find similar but progressively better models.

```powershell
python curriculum_generator.py
```

## 3. Verification Utilities (Optional)

You can run these scripts to verify the database state.

### Check Database Models

**Description:** Prints the count of models currently stored in the database.

```powershell
python check_db.py
```

### Check Similarity Edges

**Description:** Prints the count of similarity connections (edges) stored in the database.

```powershell
python check_sim.py
```

### Debug Score Distribution

**Description:** Shows statistics (min, max, avg) of the similarity scores to ensure the graph is well-formed.

```powershell
python debug_scores.py
```
