"""Option A: `uuid4` returns an architecture certificate for model code.

Kept separate from `test_arch_uid.py` so that the architecture identity and its
use as the project-wide identity can be reviewed, and adopted, independently.

    python test_uuid4_arch_uid.py        # or: pytest test_uuid4_arch_uid.py
"""

from ab.nn.util.ArchUID import arch_uid

# A minimal but complete LEMUR model: `class Net` plus the contract functions.
MODEL = """
import torch
import torch.nn as nn


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.conv = nn.Conv2d(in_shape[1], 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, out_shape[0])

    def train_setup(self, prm):
        self.criteria = (nn.CrossEntropyLoss(),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'],
                                         momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            self.optimizer.zero_grad()
            loss = self.criteria[0](self(inputs), labels)
            loss.backward()
            self.optimizer.step()

    def forward(self, x):
        x = self.pool(self.bn(self.conv(x))).flatten(1)
        return self.fc(x)
"""


def test_uuid4_uses_arch_uid_for_models_and_md5_for_everything_else():
    """Option A: `uuid4` is architecture aware for model code only.

    Every other value it is given must keep the original content hash, because
    those values are primary keys of `stat`, `run`, `tflite` and `prun`, and an
    architecture certificate cannot tell two of them apart.
    """
    import hashlib, re
    from ab.nn.util.Util import uuid4

    def md5(obj):
        return hashlib.md5(re.sub('\\s', '', str(obj)).encode()).hexdigest()

    assert uuid4(MODEL) == arch_uid(MODEL) != md5(MODEL)

    for other in (
        ['img-classification', 'cifar-10', 'acc', 'AirNext', 1],   # stat.id
        ['AirNext', 'a1b2c3d4'],                                   # nn_stat.id
        ['x.json', 'AirNext', 'android', '11', 210661000, 'fp32'], # run.id
        {'lr': 0.01, 'momentum': 0.9, 'batch': 32},                # prm.uid
        'AirNext', 'bf-v1-Pad_CenterCrop', '12345', '', 'def f(:',
        None, 123, 4.5, True, b'bytes', ('a', 'b'),
    ):
        assert uuid4(other) == md5(other), other


def test_uuid4_keeps_statistics_keys_distinct():
    """The failure mode this guards against: an empty graph hashes to one value,
    so unguarded certificates would collapse every statistics key onto it."""
    from ab.nn.util.Util import uuid4
    keys = [['img-classification', 'cifar-10', 'acc', f'model{i}', e]
            for i in range(60) for e in range(1, 6)]
    assert len({uuid4(k) for k in keys}) == len(keys)

def test_nn_id_is_one_per_architecture():
    """With `uuid4` architecture aware, `nn.id` is the architecture identifier,
    so this query is the duplicate report and must come back empty."""
    import sqlite3
    from ab.nn.util.Const import db_file
    if not db_file.exists():
        return
    con = sqlite3.connect(db_file)
    dups = con.execute("SELECT id, COUNT(*) c, GROUP_CONCAT(name) FROM nn "
                       "GROUP BY id HAVING c > 1").fetchall()
    total, distinct = con.execute("SELECT COUNT(*), COUNT(DISTINCT id) FROM nn").fetchone()
    con.close()
    assert not dups, [(d[0], d[2]) for d in dups[:5]]
    assert total == distinct, f"{total} models but {distinct} distinct ids"


def test_duplicate_model_is_not_added_twice():
    """A second file holding the same network must not create a second row.

    The database is built from the file system and anyone can add a file, so
    `code_to_db` is the only place that sees every model however it arrived. The
    stored model is kept and its name returned, so the newcomer's statistics
    attach to it. The stored row is never deleted, because every statistics table
    references nn(name) ON DELETE CASCADE.
    """
    import sqlite3, tempfile, os
    from pathlib import Path
    from ab.nn.util.db.Write import code_to_db

    renamed = MODEL.replace("class Net", "class Network").replace("Network(nn.Module)", "Net(nn.Module)")
    renamed = MODEL.replace("self.conv", "self.feature").replace("    ", "\t")
    variant = MODEL.replace("64", "128")

    with tempfile.TemporaryDirectory() as d:
        con = sqlite3.connect(os.path.join(d, "t.db"))
        cur = con.cursor()
        cur.execute("CREATE TABLE nn (name TEXT PRIMARY KEY, code TEXT NOT NULL, id TEXT NOT NULL)")
        cur.execute("CREATE INDEX idx_nn_id ON nn(id)")

        names = []
        for stem, src in (("first", MODEL), ("renamed_copy", renamed), ("real_variant", variant)):
            f = Path(d) / f"{stem}.py"
            f.write_text(src)
            names.append(code_to_db(cur, "nn", code_file=f))

        assert names[0] == "first"
        assert names[1] == "first", "the duplicate must resolve to the stored model"
        assert names[2] == "real_variant", "a genuine variant must still be added"

        rows = cur.execute("SELECT name, id FROM nn").fetchall()
        assert len(rows) == 2, rows
        assert len({i for _, i in rows}) == 2
        stored = cur.execute("SELECT code FROM nn WHERE name = 'first'").fetchone()[0]
        assert stored == MODEL, "the code already stored must be kept, not overwritten"
        con.close()


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all uuid4 cases hold")


if __name__ == "__main__":
    main()
