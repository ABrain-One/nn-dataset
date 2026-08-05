#!/bin/bash

cd "$(dirname "$0")/../.."

if [ -d .venv ]; then
    source .venv/bin/activate
fi

set -euo pipefail

# ============================================================
# Usage
# ============================================================
#
# Select exactly one CONFIG below, then run:
#
#   ./util/sh/run_layer_analysis.sh
#
# The runner:
#   - freezes the highest-numbered numeric historical JSON,
#   - uses records[0] from that file,
#   - reuses its exact supported hyperparameters, batch, and transform,
#   - trains once through 50 epochs,
#   - performs direct layer analysis at epochs
#       1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
#   - enriches the first exact historical match in existing checkpoint files,
#   - creates missing epoch files from the corresponding staged epoch,
#   - validates everything before canonical JSON/database writeback.
#
#
# ============================================================

# ============================================================
# IMAGE CLASSIFICATION
# ============================================================

# CONFIG=img-classification_cifar-10_acc_AirNet
# CONFIG=img-classification_cifar-10_acc_AirNext
# CONFIG=img-classification_cifar-10_acc_AlexNet
 #CONFIG=img-classification_cifar-10_acc_BagNet
# CONFIG=img-classification_cifar-10_acc_ComplexNet
# CONFIG=img-classification_cifar-10_acc_BayesianNet-1
# CONFIG=img-classification_cifar-10_acc_ConvNeXt
# CONFIG=img-classification_cifar-10_acc_ConvNeXtTransformer
 CONFIG=img-classification_cifar-10_acc_DPN68
# CONFIG=img-classification_cifar-10_acc_DPN107
# CONFIG=img-classification_cifar-10_acc_DPN131
# CONFIG=img-classification_cifar-10_acc_DarkNet
# CONFIG=img-classification_cifar-10_acc_DenseNet
# CONFIG=img-classification_cifar-10_acc_Diffuser
# CONFIG=img-classification_cifar-10_acc_EfficientNet
# CONFIG=img-classification_cifar-10_acc_FractalNet
# CONFIG=img-classification_cifar-10_acc_GoogLeNet
# CONFIG=img-classification_cifar-10_acc_ICNet
# CONFIG=img-classification_cifar-10_acc_InceptionV3-1
# CONFIG=img-classification_cifar-10_acc_MNASNet
# CONFIG=img-classification_cifar-10_acc_MaxVit
# CONFIG=img-classification_cifar-10_acc_MobileNetV2
# CONFIG=img-classification_cifar-10_acc_MobileNetV3
# CONFIG=img-classification_cifar-10_acc_MoE-hetero4-Alex-Dense-Air-Bag
#CONFIG=img-classification_cifar-10_acc_RegNet
# CONFIG=img-classification_cifar-10_acc_ResNet
# CONFIG=img-classification_cifar-10_acc_ShuffleNet
# CONFIG=img-classification_cifar-10_acc_SqueezeNet-1
# CONFIG=img-classification_cifar-10_acc_SwinTransformer
# CONFIG=img-classification_cifar-10_acc_UNet2D
# CONFIG=img-classification_cifar-10_acc_VGG
# CONFIG=img-classification_cifar-10_acc_VisionTransformer

# ============================================================
# IMAGE SEGMENTATION
# ============================================================

# CONFIG=img-segmentation_coco_iou_DeepLabV3-1
# CONFIG=img-segmentation_coco_iou_FCN8s
# CONFIG=img-segmentation_coco_iou_FCN16s
# CONFIG=img-segmentation_coco_iou_FCN32s-1
# CONFIG=img-segmentation_coco_iou_LRASPP
# CONFIG=img-segmentation_coco_iou_UNet-1

# ============================================================
# OBJECT DETECTION
# ============================================================

# CONFIG=obj-detection_coco_map_FasterRCNN
# CONFIG=obj-detection_coco_map_FCOS
# CONFIG=obj-detection_coco_map_RetinaNet
# CONFIG=obj-detection_coco_map_SSDLite

# ============================================================
# TEXT GENERATION
# ============================================================

# CONFIG=txt-generation_wikitext_ppl_RNN
# CONFIG=txt-generation_wikitext_ppl_LSTM

# ============================================================
# IMAGE CAPTIONING
# ============================================================

# CONFIG=img-captioning_coco_bleu4_RESNETLSTM
# CONFIG=img-captioning_coco_bleu4_ResNetTransformer

# ============================================================
# SUPER RESOLUTION
# ============================================================

# CONFIG=img-super-resolution_div2k_psnr_RLFN

python - "$CONFIG" <<'PY'
import copy
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import ab.nn.util.db.Write as DB_Write
from ab.nn.util.Loader import load_dataset
from ab.nn.util.Train import Train
from ab.nn.util.Util import default_epoch_limit_minutes, nn_mod
from ab.nn.util.db.Util import get_ab_nn_attr

CONFIG = sys.argv[1]
EPOCH_MAX = int(os.environ.get("EPOCH_MAX", "50"))


ANALYSIS_EPOCHS = tuple(
    epoch
    for epoch in range(1, EPOCH_MAX + 1)
    if epoch <= 5 or epoch % 5 == 0
)
ANALYSIS_EPOCH_SET = set(ANALYSIS_EPOCHS)

def fail(message):
    raise RuntimeError(message)

def split_config(config):
    parts = config.split("_", 3)

    if len(parts) != 4:
        fail(
            "CONFIG must have the form task_dataset_metric_model; "
            f"received {config!r}"
        )

    return tuple(parts)

def load_record_list(path):
    try:
        with path.open(encoding="utf-8") as file:
            records = json.load(file)
    except Exception as exc:
        fail(f"Cannot read valid JSON from {path}: {exc}")

    if not isinstance(records, list):
        fail(
            f"Expected a JSON list in {path}; "
            f"found {type(records).__name__}"
        )

    if not records:
        fail(f"Expected at least one record in {path}")

    if not all(isinstance(record, dict) for record in records):
        fail(f"Every entry in {path} must be a JSON object")

    return records

def numeric_json_files(config_dir):
    files = [
        path
        for path in config_dir.glob("*.json")
        if path.stem.isdigit()
    ]

    return sorted(files, key=lambda path: int(path.stem))

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()

def exact_parameters(source_record, model_name):
    supported = set(
        get_ab_nn_attr(
            f"nn.{model_name}",
            "supported_hyperparameters",
        )()
    )

    required = supported | {
        "batch",
        "transform",
    }

    missing = sorted(
        key
        for key in required
        if (
            key not in source_record
            or source_record[key] in (None, "")
        )
    )

    if missing:
        fail(
            "The selected source record is missing exact "
            "required values: "
            + ", ".join(missing)
        )

    return {
        key: copy.deepcopy(source_record[key])
        for key in sorted(required)
    }

def matches_parameters(record, parameters):
    return all(
        record.get(key) == value
        for key, value in parameters.items()
    )

def validate_direct_layer_stat(
    layer_stat,
    *,
    epoch,
    path,
):
    if not isinstance(layer_stat, dict):
        fail(
            f"Epoch {epoch} has no layer_stat object "
            f"in {path}"
        )

    expected_keys = {
        "summary",
        "layers",
        "raw_analysis",
    }

    if set(layer_stat) != expected_keys:
        fail(
            f"Epoch {epoch} has invalid direct layer_stat "
            f"keys in {path}: expected "
            f"{sorted(expected_keys)}, found "
            f"{sorted(layer_stat)}"
        )

    if any(
        str(key).isdigit()
        for key in layer_stat
    ):
        fail(
            f"Epoch {epoch} contains cumulative epoch keys "
            f"in layer_stat: {path}"
        )

    if not isinstance(
        layer_stat["summary"],
        dict,
    ):
        fail(
            f"Epoch {epoch} layer_stat.summary is not "
            f"an object in {path}"
        )

    if not isinstance(
        layer_stat["raw_analysis"],
        dict,
    ):
        fail(
            f"Epoch {epoch} layer_stat.raw_analysis is not "
            f"an object in {path}"
        )

    if (
        not isinstance(layer_stat["layers"], list)
        or not layer_stat["layers"]
    ):
        fail(
            f"Epoch {epoch} layer_stat.layers is empty "
            f"or invalid in {path}"
        )

    names = []

    for row in layer_stat["layers"]:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
        ):
            fail(
                f"Epoch {epoch} contains a layer row "
                f"without a string name in {path}"
            )

        names.append(row["name"])

    if len(names) != len(set(names)):
        fail(
            f"Epoch {epoch} contains duplicate layer names "
            f"in {path}"
        )

def run_training(
    task,
    dataset,
    metric,
    model_name,
    parameters,
    stage_dir,
):
    replay_parameters = copy.deepcopy(parameters)
    replay_parameters["epoch_max"] = EPOCH_MAX

    batch = int(replay_parameters["batch"])

    if batch < 1:
        fail(
            "batch must be a positive integer; "
            f"received {batch}"
        )

    (
        out_shape,
        minimum_accuracy,
        train_set,
        test_set,
    ) = load_dataset(
        task,
        dataset,
        replay_parameters["transform"],
    )

    trainer = Train(
        config=(
            task,
            dataset,
            metric,
            model_name,
        ),
        out_shape=out_shape,
        minimum_accuracy=minimum_accuracy,
        batch=batch,
        nn_module=nn_mod(
            "nn",
            model_name,
        ),
        task=task,
        train_dataset=train_set,
        test_dataset=test_set,
        metric=metric,
        num_workers=1,
        prm=replay_parameters,
        save_to_db=True,
        layer_analysis=True,
    )


    original_save_results = DB_Write.save_results
    original_save_layer_stat = DB_Write.save_layer_stat

    DB_Write.save_results = (
        lambda *args, **kwargs: "staging-only"
    )
    DB_Write.save_layer_stat = (
        lambda *args, **kwargs: None
    )

    try:
        return trainer.train_n_eval(
            epoch_max=EPOCH_MAX,
            epoch_limit_minutes=(
                default_epoch_limit_minutes
            ),
            save_pth_weights=False,
            save_onnx_weights=False,
            train_set=train_set,
            save_path=stage_dir,
        )
    finally:
        DB_Write.save_results = (
            original_save_results
        )
        DB_Write.save_layer_stat = (
            original_save_layer_stat
        )

def select_staged_record(
    path,
    epoch,
    parameters,
):
    records = load_record_list(path)

    matches = [
        record
        for record in records
        if matches_parameters(
            record,
            parameters,
        )
    ]

    if len(matches) != 1:
        fail(
            "Expected exactly one staged "
            f"exact-parameter record in {path}; "
            f"found {len(matches)}"
        )

    record = matches[0]

    if int(record.get("epoch_max", -1)) != EPOCH_MAX:
        fail(
            f"Staged record in {path} has "
            f"epoch_max={record.get('epoch_max')!r}; "
            f"expected {EPOCH_MAX}"
        )

    if epoch in ANALYSIS_EPOCH_SET:
        validate_direct_layer_stat(
            record.get("layer_stat"),
            epoch=epoch,
            path=path,
        )
    elif "layer_stat" in record:
        fail(
            f"Non-checkpoint epoch {epoch} unexpectedly "
            f"contains layer_stat in {path}"
        )

    return copy.deepcopy(record)

def layer_table(layer_stat, epoch):
    table = {}

    for layer in layer_stat["layers"]:
        row = copy.deepcopy(layer)
        name = row.pop("name")

        if name in table:
            fail(
                f"Duplicate layer name {name!r} "
                f"at epoch {epoch}"
            )

        table[name] = row

    return table

def validate_db_payload(record, epoch):
    if (
        not isinstance(record.get("uid"), str)
        or not record["uid"]
    ):
        fail(
            f"Epoch {epoch} record has no non-empty uid "
            "for database association"
        )

    if record.get("transform") in (None, ""):
        fail(
            f"Epoch {epoch} record has no transform "
            "for database association"
        )

    payload = {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key != "layer_stat"
    }

    for key, value in payload.items():
        if key == "train_stat":
            if (
                value is not None
                and not isinstance(value, dict)
            ):
                fail(
                    f"Epoch {epoch} train_stat "
                    "is not an object"
                )
        elif isinstance(value, (dict, list)):
            fail(
                f"Epoch {epoch} DB field {key!r} "
                "is unexpectedly nested"
            )

    return payload

def write_prepared_temp(path, records):
    temporary = path.with_name(
        f".{path.name}.layer-analysis-"
        f"{os.getpid()}.tmp"
    )

    if temporary.exists():
        temporary.unlink()

    with temporary.open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            indent=4,
            ensure_ascii=False,
        )
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())

    if load_record_list(temporary) != records:
        temporary.unlink(missing_ok=True)

        fail(
            "Prepared JSON failed read-back "
            f"verification: {temporary}"
        )

    return temporary

def sqlite_backup(
    source_path,
    destination_path,
):
    destination_path.unlink(missing_ok=True)

    source = sqlite3.connect(
        str(source_path)
    )
    destination = sqlite3.connect(
        str(destination_path)
    )

    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()

def restore_sqlite_backup(
    backup_path,
    destination_path,
):
    source = sqlite3.connect(
        str(backup_path)
    )
    destination = sqlite3.connect(
        str(destination_path)
    )

    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()

def restore_json_files(
    prepared,
    rollback_files,
    existed_before,
):
    for path in prepared:
        if existed_before[path]:
            restore_temp = path.with_name(
                f".{path.name}."
                f"layer-analysis-restore-"
                f"{os.getpid()}.tmp"
            )

            shutil.copy2(
                rollback_files[path],
                restore_temp,
            )

            os.replace(
                restore_temp,
                path,
            )
        else:
            path.unlink(missing_ok=True)

task, dataset, metric, model_name = split_config(
    CONFIG
)

config_dir = (
    Path("ab/nn/stat/train")
    / CONFIG
)

if not config_dir.is_dir():
    fail(
        "Historical configuration directory "
        f"does not exist: {config_dir}"
    )

original_paths = numeric_json_files(
    config_dir
)

if not original_paths:
    fail(
        "No numeric historical JSON files "
        f"found in {config_dir}"
    )

# Freeze source selection before staging creates anything.
source_path = original_paths[-1]

original_records = {
    path: load_record_list(path)
    for path in original_paths
}

original_hashes = {
    path: sha256_file(path)
    for path in original_paths
}

# records[0] is intentionally selected. The historical files
# are already ordered with the best accuracy first.
source_record = copy.deepcopy(
    original_records[source_path][0]
)

parameters = exact_parameters(
    source_record,
    model_name,
)

if not isinstance(
    source_record.get("accuracy"),
    (int, float),
):
    fail(
        f"First record in {source_path} "
        "has no numeric accuracy"
    )

stage_root = Path(
    "out/layer-analysis-stage"
)

stage_dir = (
    stage_root
    / (
        f"{CONFIG}-"
        f"{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{os.getpid()}"
    )
)

stage_dir.mkdir(
    parents=True,
    exist_ok=False,
)

rollback_dir = (
    stage_dir
    / "rollback"
)

rollback_dir.mkdir()

print("Selected source")
print("===============")
print(f"CONFIG: {CONFIG}")
print(f"source JSON: {source_path}")
print("source record index: 0")
print(
    f"source accuracy: "
    f"{source_record['accuracy']}"
)
print("exact parameters:")
print(
    json.dumps(
        parameters,
        indent=2,
        ensure_ascii=False,
    )
)
print(
    f"staging directory: "
    f"{stage_dir}"
)
print(f"training epochs: 1-{EPOCH_MAX}")
print(
    f"analysis epochs: "
    f"{list(ANALYSIS_EPOCHS)}"
)

print()

run_training(
    task,
    dataset,
    metric,
    model_name,
    parameters,
    stage_dir,
)

# Validate a complete direct per-epoch staged run.
staged_records = {}

for epoch in range(
    1,
    EPOCH_MAX + 1,
):
    staged_path = (
        stage_dir
        / f"{epoch}.json"
    )

    if not staged_path.is_file():
        fail(
            f"Missing staged epoch file: "
            f"{staged_path}"
        )

    staged_records[epoch] = (
        select_staged_record(
            staged_path,
            epoch,
            parameters,
        )
    )

# Refuse writeback if canonical history changed during training.
current_paths = numeric_json_files(
    config_dir
)

if current_paths != original_paths:
    fail(
        "Numeric historical JSON file set changed "
        "during training; refusing writeback"
    )

for path, expected_hash in original_hashes.items():
    if sha256_file(path) != expected_hash:
        fail(
            "Historical JSON changed during training; "
            f"refusing writeback: {path}"
        )

prepared = {}
db_jobs = []
existed_before = {}
rollback_files = {}

for epoch in range(
    1,
    EPOCH_MAX + 1,
):
    canonical_path = (
        config_dir
        / f"{epoch}.json"
    )

    existed_before[canonical_path] = (
        canonical_path.is_file()
    )

    if canonical_path.is_file():
        # Existing non-checkpoint history is preserved
        # byte-for-byte.
        if epoch not in ANALYSIS_EPOCH_SET:
            continue

        records = copy.deepcopy(
            original_records[canonical_path]
        )

        match_index = next(
            (
                index
                for index, record in enumerate(records)
                if matches_parameters(
                    record,
                    parameters,
                )
            ),
            None,
        )

        if match_index is None:
            fail(
                f"Existing checkpoint {canonical_path} "
                "has no exact-parameter record; refusing "
                "to append, sort, or guess"
            )

        before = copy.deepcopy(
            records[match_index]
        )

        records[match_index]["layer_stat"] = (
            copy.deepcopy(
                staged_records[epoch]["layer_stat"]
            )
        )

        after_without_layer = {
            key: value
            for key, value
            in records[match_index].items()
            if key != "layer_stat"
        }

        before_without_layer = {
            key: value
            for key, value
            in before.items()
            if key != "layer_stat"
        }

        if (
            after_without_layer
            != before_without_layer
        ):
            fail(
                "Non-layer historical data changed "
                f"while preparing {canonical_path}"
            )

        prepared[canonical_path] = records

        db_jobs.append(
            (
                epoch,
                copy.deepcopy(
                    records[match_index]
                ),
            )
        )
    else:
        # The whole corresponding staged epoch record is
        # used only when the canonical epoch file is missing.
        records = [
            copy.deepcopy(
                staged_records[epoch]
            )
        ]

        prepared[canonical_path] = records

        db_jobs.append(
            (
                epoch,
                copy.deepcopy(records[0]),
            )
        )

# Validate and materialize every replacement before touching
# the database or canonical files.
prepared_temps = {}

for path, records in sorted(
    prepared.items(),
    key=lambda item: int(item[0].stem),
):
    if path.is_file():
        backup = (
            rollback_dir
            / path.name
        )

        shutil.copy2(
            path,
            backup,
        )

        rollback_files[path] = backup

    prepared_temps[path] = (
        write_prepared_temp(
            path,
            records,
        )
    )

# Ensure every database payload is valid before the first
# database mutation.
validated_db_jobs = [
    (
        epoch,
        record,
        validate_db_payload(
            record,
            epoch,
        ),
    )
    for epoch, record in db_jobs
]

db_path = Path(
    DB_Write.db_file
)

if not db_path.is_file():
    fail(
        f"Database file does not exist: "
        f"{db_path}"
    )

db_backup = (
    stage_dir
    / "ab.nn.db.before-writeback"
)

# Hold the repository database-writer lock through backup,
# database update, verification, canonical replacement, and
# rollback if anything fails.
with DB_Write._db_write_lock():
    sqlite_backup(
        db_path,
        db_backup,
    )

    db_expectations = []

    try:
        for (
            epoch,
            record,
            payload,
        ) in validated_db_jobs:
            stat_id = DB_Write.save_results(
                (
                    task,
                    dataset,
                    metric,
                    model_name,
                    epoch,
                ),
                payload,
            )

            if (
                not isinstance(stat_id, str)
                or not stat_id
            ):
                fail(
                    "Database save returned no "
                    f"stat_id for epoch {epoch}"
                )

            expected_layers = 0

            if epoch in ANALYSIS_EPOCH_SET:
                table = layer_table(
                    record["layer_stat"],
                    epoch,
                )

                expected_layers = len(table)

                DB_Write.save_layer_stat(
                    epoch,
                    table,
                    stat_id,
                    metric,
                )

            db_expectations.append(
                (
                    epoch,
                    stat_id,
                    record["uid"],
                    expected_layers,
                )
            )

        connection = sqlite3.connect(
            str(db_path)
        )

        try:
            for (
                epoch,
                stat_id,
                expected_uid,
                expected_layers,
            ) in db_expectations:
                row = connection.execute(
                    """
                    SELECT prm
                    FROM stat
                    WHERE id = ?
                    """,
                    (stat_id,),
                ).fetchone()

                if (
                    row is None
                    or row[0] != expected_uid
                ):
                    fail(
                        f"Epoch {epoch} DB association "
                        f"mismatch: stat.id={stat_id}, "
                        "expected "
                        f"stat.prm={expected_uid!r}"
                    )

                if expected_layers:
                    layer_count = (
                        connection.execute(
                            """
                            SELECT COUNT(*)
                            FROM layer_stat
                            WHERE stat_id = ?
                            """,
                            (stat_id,),
                        ).fetchone()[0]
                    )

                    per_layer_count = (
                        connection.execute(
                            """
                            SELECT COUNT(*)
                            FROM per_layer_stat
                            WHERE stat_id = ?
                            """,
                            (stat_id,),
                        ).fetchone()[0]
                    )

                    if (
                        layer_count
                        != expected_layers
                        or per_layer_count
                        != expected_layers
                    ):
                        fail(
                            f"Epoch {epoch} DB layer-row "
                            "mismatch for "
                            f"stat.id={stat_id}: "
                            f"expected {expected_layers}, "
                            f"layer_stat={layer_count}, "
                            "per_layer_stat="
                            f"{per_layer_count}"
                        )
        finally:
            connection.close()

        for path, temporary in sorted(
            prepared_temps.items(),
            key=lambda item: int(
                item[0].stem
            ),
        ):
            os.replace(
                temporary,
                path,
            )

        for (
            path,
            expected_records,
        ) in prepared.items():
            if (
                load_record_list(path)
                != expected_records
            ):
                fail(
                    "Canonical JSON read-back "
                    f"verification failed: {path}"
                )

    except BaseException:
        restore_json_files(
            prepared,
            rollback_files,
            existed_before,
        )

        restore_sqlite_backup(
            db_backup,
            db_path,
        )

        for temporary in (
            prepared_temps.values()
        ):
            temporary.unlink(
                missing_ok=True
            )

        raise

created_epochs = [
    int(path.stem)
    for path in prepared
    if not existed_before[path]
]

updated_epochs = [
    int(path.stem)
    for path in prepared
    if existed_before[path]
]


shutil.rmtree(
    stage_dir
)

# Remove the staging root only if no other run directories remain.
try:
    stage_root.rmdir()
except OSError:
    pass

print()
print("Layer-analysis writeback completed")
print("==================================")
print(
    f"source: {source_path} records[0]"
)
print(
    f"trained epochs: 1-{EPOCH_MAX}"
)
print(
    f"analysis epochs: "
    f"{list(ANALYSIS_EPOCHS)}"
)
print(
    "updated existing checkpoint files: "
    f"{sorted(updated_epochs)}"
)
print(
    "created missing epoch files: "
    f"{sorted(created_epochs)}"
)
print(
    f"database: {db_path}"
)
print(
    "temporary staging and rollback files removed: "
    f"{stage_dir}"
)
PY