# Run

## 1. Prerequisites

- Python 3.10+ recommended
- Use a virtual environment

## 2. Clean install from fork

```bash
rm -rf db
pip uninstall -y nn-dataset
pip install --no-cache-dir git+https://github.com/i-am-manishasamal/nn-dataset --extra-index-url https://download.pytorch.org/whl/cu126
```

## 3. Run verification from NNGPT

```bash
python test.py
```

## 4. Optional benchmark timing

```bash
for i in 1 2 3; do rm -rf db && python test.py; done
```

## 5. Notes

- Run `rm -rf db` before each clean verification run.
- Run `python test.py` from inside the NNGPT project.
