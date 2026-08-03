# LEMUR Query Tool — User Guide

`lemur_query.py` is a standalone read-only extraction tool for the LEMUR NN Dataset. It allows users to explore available tasks, view model performance, extract best accuracy results, and download model source code — all from the command line.

> **Safe to use:** This script never modifies, deletes, or uploads anything. It only reads from the database.

---

## Requirements

Before using the tool, make sure you have the following set up:

- Python 3.10 or higher
- The `nn-dataset` repository cloned locally
- The virtual environment (`lemur_env`) activated
- The database file present at `db/ab.nn.db`

---

## Setup

Open a terminal and navigate to the repository folder:

```bash
cd ~/CV\ Praktikum/nn-dataset
```

Activate the virtual environment:

```bash
source lemur_env/bin/activate
```

You are now ready to run the tool.

---

## What Can This Tool Do?

| Feature | Description |
|---|---|
| List tasks | See all available task categories in the dataset |
| Task summary | View all models and datasets for a task, with their best accuracy |
| Best accuracy | See the best accuracy each model has ever achieved, grouped by dataset |
| Extract code | Download the Python source code of one or more models |
| Export to Excel | Save any table as a `.xlsx` file with proper columns |
| Export to CSV | Save any table as a `.csv` file |
| Export to TXT | Save any table as a plain text file |

---

## Step-by-Step Guide

### Step 1 — Find out what tasks are available

If you are not sure what tasks exist in the database, run the tool with no arguments:

```bash
python lemur_query.py
```

**Example output:**
```
Available tasks (9):

  img-captioning
  img-classification
  img-denoising
  img-segmentation
  img-sr
  img-super-resolution
  obj-detection
  txt-generation
  txt-image
```

Use one of these exact task names in the next steps.

---

### Step 2 — View models and best accuracy for a specific task

To see all models and datasets for a task, along with the maximum accuracy each model achieved:

```bash
python lemur_query.py --task "img-classification"
```

**Example output:**
```
Task: img-classification
──────────────────────────────────────────────────────────────────────
      Model    Dataset  Max Accuracy
    AlexNet   cifar10      0.823400
    AlexNet  cifar100      0.512300
   ResNet18   cifar10      0.941200
   ResNet18  cifar100      0.731500
```

---

### Step 3 — Save the results to a file

You can save the output in three different file formats. Choose the one that suits you best.

**Save as Excel (recommended — opens directly in Excel with proper columns):**
```bash
python lemur_query.py --task "img-classification" --csv results.xlsx
```

**Save as CSV (for Google Sheets or other tools):**
```bash
python lemur_query.py --task "img-classification" --csv results.csv
```

**Save as plain text (for simple sharing or reading):**
```bash
python lemur_query.py --task "img-classification" --txt results.txt
```

**Save in multiple formats at once:**
```bash
python lemur_query.py --task "img-classification" --csv results.xlsx --txt results.txt
```

> The terminal output is always shown regardless of which file format you choose.

---

### Step 4 — View best accuracy per model grouped by dataset

To see the best accuracy ever achieved by each model on each dataset (across all tasks):

```bash
python lemur_query.py --best-accuracy
```

To restrict this to a specific task:

```bash
python lemur_query.py --best-accuracy --task "img-classification"
```

To save the results:

```bash
python lemur_query.py --best-accuracy --csv best_accuracy.xlsx
python lemur_query.py --best-accuracy --task "img-classification" --csv best_accuracy.xlsx --txt best_accuracy.txt
```

---

### Step 5 — Extract model source code

To download the Python source code of a model stored in the database, provide a model name prefix:

```bash
python lemur_query.py --code AlexNet
```

This saves a `.py` file for every model whose name starts with `AlexNet` (including generated variants like `AlexNet-abc123...`) in your current directory.

**Extract multiple models at once** by separating prefixes with a comma:

```bash
python lemur_query.py --code AlexNet,ResNet
```

**Save the code files to a specific folder:**

```bash
python lemur_query.py --code AlexNet --out-dir ./my_models
```

---

## Quick Reference

| What you want | Command |
|---|---|
| See all available tasks | `python lemur_query.py` |
| Models + accuracy for a task | `python lemur_query.py --task "img-classification"` |
| Save task results as Excel | `python lemur_query.py --task "img-classification" --csv results.xlsx` |
| Save task results as CSV | `python lemur_query.py --task "img-classification" --csv results.csv` |
| Save task results as TXT | `python lemur_query.py --task "img-classification" --txt results.txt` |
| Best accuracy (all tasks) | `python lemur_query.py --best-accuracy` |
| Best accuracy for one task | `python lemur_query.py --best-accuracy --task "img-classification"` |
| Save best accuracy as Excel | `python lemur_query.py --best-accuracy --csv best.xlsx` |
| Get model source code | `python lemur_query.py --code AlexNet` |
| Get code for multiple models | `python lemur_query.py --code AlexNet,ResNet` |
| Save code to a specific folder | `python lemur_query.py --code AlexNet --out-dir ./my_models` |

---

## Output File Formats

| Format | Extension | Best used for |
|---|---|---|
| Excel | `.xlsx` | Opening in Microsoft Excel or LibreOffice Calc with proper columns |
| CSV | `.csv` | Google Sheets, data analysis tools, importing into other software |
| Plain text | `.txt` | Simple reading, sharing, or pasting into documents |

---

## Notes

- The tool always prints results to the terminal. File saving is always optional.
- Task names must match exactly as shown by `python lemur_query.py` (e.g. `img-classification`, not `image classification`).
- Model code is only available for models where source code was stored in the database. Generated variant models may not have code available.
- The `--csv` and `--txt` flags can be used together in the same command to save both formats at once.
