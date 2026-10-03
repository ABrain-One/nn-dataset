# Final 5-Epoch Acceptance Candidate

- Eligible models: **1085**
- Exact measured records copied: **9398**
- Epoch coverage: every selected model has real epochs **1, 2, 3, 4, 5**
- Source hash matches: **1085/1085**
- Required hyperparameters present in every record: **['batch', 'lr', 'momentum', 'transform', 'weight_decay']**
- Statistics are copied only from recovered measured JSON records; no records were fabricated, copied across epochs, interpolated, or estimated.
- Historical source artifacts were not modified.
- Submission-attributable underscore LR model/statistics files are excluded; the candidate contains only hyphenated names.

## Trial-count distribution

| Trial records per model | Models |
|---:|---:|
| 5 | 502 |
| 6 | 2 |
| 8 | 3 |
| 9 | 1 |
| 10 | 369 |
| 11 | 3 |
| 12 | 4 |
| 13 | 1 |
| 14 | 3 |
| 15 | 184 |
| 17 | 2 |
| 19 | 1 |
| 20 | 9 |
| 24 | 1 |

## Required separate groups

- Complete 5-epoch models with fewer than 3 trials: **0** (`complete_5epoch_fewer_than_3_trials.json`)
- Complete 5-epoch models with at least 3 trials: **1085** (`complete_5epoch_at_least_3_trials.json`)
- Complete 5-epoch models with 10 trials: **369** (`complete_5epoch_10_trials.json`)
- Models with only epoch-1 data: **1401** (`models_with_only_epoch1.json`)
- Models with partial multi-epoch data: **506** (`models_with_partial_multi_epoch.json`)
- Models with partial or epoch-1 data combined: **1907** (`partial_or_epoch1_models.json`)

## Validation status

Importer and complete-repository test results:

- Complete repository tests: **11 passed, 0 failed, 0 errors (OK)**.
- Candidate-attributable importer warnings: **0**.
- Unrelated repository importer warnings: **3,000**; these are from pre-existing historical statistics outside `final_5epoch_models/` and `final_5epoch_statistics/`, primarily underscore LR directories and other unrelated records.
- The final candidate contains no `lr_####.py` files and no `img-classification_cifar-10_acc_lr_####` directories.
