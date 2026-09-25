"""What the architecture UID is, and is not, invariant to.

Each case is a controlled edit to one reference head. `SAME` means the edit
should not change architectural identity; `DIFF` means it must. A row that does
not behave as marked is a property the certificate does not actually have.

Usage:
    python test_arch_uid.py              # verbose, one line per case
    python -m unittest test_arch_uid -v
    pytest test_arch_uid.py
"""

from __future__ import annotations

import re
import unittest

from ab.nn.util.ArchUID import arch_similarity, arch_uid

BASE = """
class Head(nn.Module):
    def __init__(self, in_channels, num_classes, prm):
        super().__init__()
        p = float(prm['dropout'])
        self.reduce = nn.Conv2d(in_channels, 128, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))
"""

LOOP_FORM = """
class Head(nn.Module):
    def __init__(self, in_channels, num_classes, prm):
        super().__init__()
        layers = []
        for _ in range(3):
            layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)
"""

UNROLLED_FORM = """
class Head(nn.Module):
    def __init__(self, in_channels, num_classes, prm):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.body(x)
"""

TRIP_COUNT_FORM = LOOP_FORM.replace("range(3)", "range(7)")

SGD_SETUP = """
class Net(nn.Module):
    def train_setup(self, prm):
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'])
"""
ADAM_SETUP = SGD_SETUP.replace("SGD", "AdamW")

ORDER_A = """
class Net(nn.Module):
    def learn(self, train_data):
        for inputs, labels in train_data:
            self.optimizer.zero_grad()
            loss = self.criteria(self(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
"""
ORDER_B = """
class Net(nn.Module):
    def learn(self, train_data):
        for inputs, labels in train_data:
            self.optimizer.zero_grad()
            loss = self.criteria(self(inputs), labels)
            loss.backward()
            self.optimizer.step()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
"""


# Two blocks in SEPARATE classes that both name their activation `self.act`.
# A single shared attribute namespace merges them, so exchanging the two
# activations only reorders one bucket and the graph does not change.
SWAP_A = """
class Down(nn.Module):
    def __init__(self, f):
        self.conv = nn.Conv2d(f, f, 3, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.act(self.conv(x))

class Up(nn.Module):
    def __init__(self, f):
        self.up = nn.ConvTranspose2d(f, f, 2, stride=2)
        self.act = nn.ELU(0.2, inplace=True)
    def forward(self, y, skip):
        return self.act(self.up(y)) + skip
"""
SWAP_B = (SWAP_A.replace("self.act = nn.LeakyReLU(0.2, inplace=True)", "@1@")
                .replace("self.act = nn.ELU(0.2, inplace=True)",
                         "self.act = nn.LeakyReLU(0.2, inplace=True)")
                .replace("@1@", "self.act = nn.ELU(0.2, inplace=True)"))

DEF_LIVE = """
class Block(nn.Module):
    def __init__(self, c, factor=4):
        super().__init__()
        self.conv = nn.Conv2d(c, c // factor, 1)
    def forward(self, x):
        return self.conv(x)
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.b = Block(64)
    def forward(self, x):
        return self.b(x)
"""
DEF_DEAD = DEF_LIVE.replace("Block(64)", "Block(64, factor=4)")
MOD_CONST = """
MOM = 0.1
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.c = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64, momentum=MOM)
    def forward(self, x):
        return self.bn(self.c(x))
"""
SUB_STORE = """
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        ws = [32, 64]
        ws[0] = 16
        self.c1 = nn.Conv2d(3, ws[0], 3)
        self.c2 = nn.Conv2d(ws[0], ws[1], 3)
    def forward(self, x):
        return self.c2(self.c1(x))
"""
PRM_STORE = """
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.c = nn.Conv2d(3, 64, 3)
    def train_setup(self, prm):
        prm['lr'] = 0.001
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'])
    def forward(self, x):
        return self.c(x)
"""
HELPER = """
class Helper(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
    def forward(self, x):
        return self.conv(x)
class Other(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Linear(c, c)
    def forward(self, x):
        return self.fc(x)
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.h = Helper(64)
    def forward(self, x):
        return self.h(x)
"""
HELPER_FN = """
def make_act():
    return nn.ReLU()
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.c = nn.Conv2d(3, 64, 3)
        self.act = make_act()
    def forward(self, x):
        return self.act(self.c(x))
"""

CASES = [
    ("SAME", "attribute rename (self.reduce -> self.proj)",
     BASE, BASE.replace("self.reduce", "self.proj")),
    ("SAME", "local temporary rename (y = ... -> z = ...)",
     "class H(nn.Module):\n    def forward(self, x):\n        y = self.fc(x)\n        return y\n",
     "class H(nn.Module):\n    def forward(self, x):\n        z = self.fc(x)\n        return z\n"),
    ("SAME", "function PARAMETER rename (forward(self, x) -> forward(self, inp))",
     "class H(nn.Module):\n    def forward(self, x):\n        return self.fc(x)\n",
     "class H(nn.Module):\n    def forward(self, inp):\n        return self.fc(inp)\n"),
    ("SAME", "whitespace / trailing-paren restyle",
     BASE, BASE.replace("bias=False)", "bias=False,\n        )")),
    ("SAME", "import alias (nn.Conv2d -> torch.nn.Conv2d)",
     BASE, BASE.replace("nn.Conv2d", "torch.nn.Conv2d")),
    ("SAME", "comment added",
     BASE, BASE.replace("super().__init__()", "super().__init__()  # init")),
    ("SAME", "super(Cls, self).__init__() vs super().__init__()",
     "class B(nn.Module):\n    def __init__(self):\n        super(B, self).__init__()\n"
     "        self.c = nn.Conv2d(3, 8, 1)\n",
     "class B(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
     "        self.c = nn.Conv2d(3, 8, 1)\n"),
    ("SAME", "type annotations added",
     "class B(nn.Module):\n    def f(self, x, y=1):\n        return self.c(x)\n",
     "class B(nn.Module):\n    def f(self, x: torch.Tensor, y: int = 1) -> torch.Tensor:\n"
     "        return self.c(x)\n"),
    ("SAME", "docstring added / reworded",
     "class H(nn.Module):\n    def forward(self, x):\n        return self.fc(x)\n",
     "class H(nn.Module):\n    def forward(self, x):\n        \"\"\"Run the head.\"\"\"\n"
     "        return self.fc(x)\n"),
    ("SAME", "constant folding (Linear(9216,..) vs Linear(256*6*6,..))",
     "class H(nn.Module):\n    def __init__(self):\n        self.fc = nn.Linear(9216, 4096)\n",
     "class H(nn.Module):\n    def __init__(self):\n        self.fc = nn.Linear(256*6*6, 4096)\n"),

    ("DIFF", "channel width 128 -> 256",
     BASE, BASE.replace("128", "256")),
    ("DIFF", "kernel size 1 -> 3",
     BASE, BASE.replace("kernel_size=1", "kernel_size=3")),
    ("DIFF", "op swapped (ReLU -> GELU)",
     BASE, BASE.replace("nn.ReLU(inplace=True)", "nn.GELU()")),
    ("DIFF", "layer removed (no BatchNorm)",
     BASE, BASE.replace("        x = self.bn(x)\n", "")),
    ("DIFF", "backbone choice inside a CHAINED call  .to(device)",
     "class Net(nn.Module):\n    def __init__(self, device):\n"
     "        self.b = TorchVision('mobilenet_v3_large', in_channels=3).to(device)\n",
     "class Net(nn.Module):\n    def __init__(self, device):\n"
     "        self.b = TorchVision('shufflenet_v2_x1_0', in_channels=3).to(device)\n"),
    ("DIFF", "backbone choice",
     BASE + "\n_backbone_choice('mnasnet0_75')\n",
     BASE + "\n_backbone_choice('efficientnet_b0')\n"),
    ("DIFF", "optimizer SGD -> AdamW  (train_setup)",
     SGD_SETUP, ADAM_SETUP),
    ("DIFF", "grad-clip before vs after step  (learn)",
     ORDER_A, ORDER_B),
    ("DIFF", "loop trip count 3 -> 7",
     LOOP_FORM, TRIP_COUNT_FORM),

    ("SAME", "loop-built vs unrolled Sequential (3 identical convs)",
     LOOP_FORM, UNROLLED_FORM),
    ("SAME", "comprehension-built vs unrolled Sequential",
     "class H(nn.Module):\n    def __init__(self):\n        self.body = nn.Sequential("
     "*[nn.Conv2d(128, 128, 3) for _ in range(3)])\n",
     "class H(nn.Module):\n    def __init__(self):\n        self.body = nn.Sequential("
     "nn.Conv2d(128, 128, 3), nn.Conv2d(128, 128, 3), nn.Conv2d(128, 128, 3))\n"),
    ("SAME", "loop variable used in widths vs written-out widths",
     "class H(nn.Module):\n    def __init__(self):\n        L = []\n"
     "        for i in range(3):\n            L.append(nn.Conv2d(32*i+32, 64, 1))\n"
     "        self.body = nn.Sequential(*L)\n",
     "class H(nn.Module):\n    def __init__(self):\n        self.body = nn.Sequential("
     "nn.Conv2d(32, 64, 1), nn.Conv2d(64, 64, 1), nn.Conv2d(96, 64, 1))\n"),
    ("DIFF", "activations SWAPPED between two equal-width blocks",
     "class Net(nn.Module):\n    def __init__(self):\n"
     "        self.enc = nn.Sequential(nn.Conv2d(64, 64, 3), nn.LeakyReLU(0.2))\n"
     "        self.dec = nn.Sequential(nn.Conv2d(64, 64, 3), nn.ELU(1.0))\n"
     "    def forward(self, x):\n        return self.dec(self.enc(x))\n",
     "class Net(nn.Module):\n    def __init__(self):\n"
     "        self.enc = nn.Sequential(nn.Conv2d(64, 64, 3), nn.ELU(1.0))\n"
     "        self.dec = nn.Sequential(nn.Conv2d(64, 64, 3), nn.LeakyReLU(0.2))\n"
     "    def forward(self, x):\n        return self.dec(self.enc(x))\n"),
    ("DIFF", "activations swapped across a SYMMETRIC skip (a + b)",
     "class Net(nn.Module):\n    def __init__(self):\n"
     "        self.e = nn.Sequential(nn.Conv2d(64, 64, 3), nn.LeakyReLU(0.2))\n"
     "        self.d = nn.Sequential(nn.Conv2d(64, 64, 3), nn.ELU(1.0))\n"
     "    def forward(self, x):\n        return self.e(x) + self.d(x)\n",
     "class Net(nn.Module):\n    def __init__(self):\n"
     "        self.e = nn.Sequential(nn.Conv2d(64, 64, 3), nn.ELU(1.0))\n"
     "        self.d = nn.Sequential(nn.Conv2d(64, 64, 3), nn.LeakyReLU(0.2))\n"
     "    def forward(self, x):\n        return self.e(x) + self.d(x)\n"),
    ("DIFF", "activations swapped between two SEPARATE classes (same attr name)",
     SWAP_A, SWAP_B),
    ("DIFF", "layer ORDER swapped inside the stack",
     "class H(nn.Module):\n    def __init__(self):\n        L = []\n"
     "        L.append(nn.Conv2d(3, 8, 1))\n        L.append(nn.BatchNorm2d(8))\n"
     "        self.body = nn.Sequential(*L)\n",
     "class H(nn.Module):\n    def __init__(self):\n        L = []\n"
     "        L.append(nn.BatchNorm2d(8))\n        L.append(nn.Conv2d(3, 8, 1))\n"
     "        self.body = nn.Sequential(*L)\n"),
    # --- 2026-09-08: blind spots found while auditing the merge ------------
    ("DIFF", "LIVE default argument  factor=4 vs 8, never passed at the call site",
     DEF_LIVE, DEF_LIVE.replace("factor=4", "factor=8")),
    ("SAME", "DEAD default argument  factor=4 vs 8, every call site passes it",
     DEF_DEAD, DEF_DEAD.replace("c, factor=4)", "c, factor=8)")),
    ("DIFF", "module-level constant read in a method  MOM 0.1 vs 0.001",
     MOD_CONST, MOD_CONST.replace("MOM = 0.1", "MOM = 0.001")),
    ("SAME", "unused module-level constant added",
     MOD_CONST, MOD_CONST.replace("MOM = 0.1", "MOM = 0.1\nUNUSED = 7")),
    ("DIFF", "subscript store on a list  ws[0] = 16 vs absent",
     SUB_STORE, SUB_STORE.replace("        ws[0] = 16\n", "")),
    ("SAME", "subscript store == written-out list  [32,64]; ws[0]=16 vs [16,64]",
     SUB_STORE, SUB_STORE.replace("[32, 64]", "[16, 64]").replace("        ws[0] = 16\n", "")),
    ("DIFF", "hyperparameter override in code  prm['lr'] = 0.001 vs absent",
     PRM_STORE, PRM_STORE.replace("        prm['lr'] = 0.001\n", "")),
    ("SAME", "helper CLASS renamed consistently  Helper -> Widget",
     HELPER, HELPER.replace("Helper", "Widget")),
    ("DIFF", "different helper class used  Helper (conv) vs Other (linear)",
     HELPER, HELPER.replace("self.h = Helper(64)", "self.h = Other(64)")),
    ("SAME", "module-level helper FUNCTION renamed consistently",
     HELPER_FN, HELPER_FN.replace("make_act", "build_activation")),
    ("DIFF", "module-level helper function body changed  ReLU -> LeakyReLU",
     HELPER_FN, HELPER_FN.replace("nn.ReLU()", "nn.LeakyReLU()")),
    ("SAME", "local callee: keyword vs positional  Block(64, factor=4) vs Block(64, 4)",
     DEF_DEAD, DEF_DEAD.replace("Block(64, factor=4)", "Block(64, 4)")),
    ("SAME", "local callee: parameter renamed at def AND call  factor -> f",
     DEF_DEAD, DEF_DEAD.replace("factor", "f")),
    ("DIFF", "library call keyword still matters  Conv2d(kernel_size=3) vs (stride=3)",
     "class Net(nn.Module):\n    def __init__(self):\n        self.c = nn.Conv2d(3, 8, kernel_size=3)\n",
     "class Net(nn.Module):\n    def __init__(self):\n        self.c = nn.Conv2d(3, 8, stride=3)\n"),
    ("SAME", "self.method(x, probe=True): keyword param renamed at def AND call",
     "class Net(nn.Module):\n    def forward(self, x, probe=False):\n        return x if probe else x * 2\n"
     "    def shape(self, d):\n        return self.forward(d, probe=True)\n",
     "class Net(nn.Module):\n    def forward(self, x, is_probing=False):\n        return x if is_probing else x * 2\n"
     "    def shape(self, d):\n        return self.forward(d, is_probing=True)\n"),
]


CHAIN = """
class H(nn.Module):
    def __init__(self, c, n):
        self.a = nn.Conv2d(c, 64, 3)
        self.b = nn.Conv2d(64, 64, 3)
        self.d = nn.BatchNorm2d(64)
    def forward(self, x):
        return self.d(self.b(self.a(x)))
"""
RESIDUAL = """
class H(nn.Module):
    def __init__(self, c, n):
        self.a = nn.Conv2d(c, 64, 3)
        self.b = nn.Conv2d(64, 64, 3)
        self.d = nn.BatchNorm2d(64)
    def forward(self, x):
        h = self.a(x)
        return self.d(self.b(h) + h)
"""

# The graded measure must agree with the certificate at the endpoints and must
# order intermediate cases sensibly. Each entry: (label, a, b, lo, hi).
SIM_CASES = [
    ("identical code == 1.0", BASE, BASE, 0.999, 1.0),
    ("pure rename == 1.0", BASE, BASE.replace("self.reduce", "self.proj"), 0.999, 1.0),
    ("one layer removed stays high", BASE,
     BASE.replace("        x = self.bn(x)\n", ""), 0.55, 0.99),
    ("chain vs residual: related, not same", CHAIN, RESIDUAL, 0.45, 0.95),
    ("width change is a real drop", BASE, BASE.replace("128", "512"), 0.05, 0.75),
    ("unrelated network is low", BASE,
     "class H(nn.Module):\n    def forward(self, x):\n        return x.mean()\n",
     0.0, 0.35),
]


# --------------------------------------------------------------------------
# unittest: one generated test method per case, so every row is reported
# individually (with its label as the description under -v).
# --------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"\W+", "_", text).strip("_").lower()[:60]


def _make_uid_test(expect, label, a, b):
    def test(self):
        ua, ub = arch_uid(a), arch_uid(b)
        if expect == "SAME":
            self.assertEqual(ua, ub, f"expected SAME, got DIFF: {label}")
        else:
            self.assertNotEqual(ua, ub, f"expected DIFF, got SAME: {label}")
    test.__doc__ = f"[{expect}] {label}"
    return test


def _make_sim_test(label, a, b, lo, hi):
    def test(self):
        v = arch_similarity(a, b)
        self.assertTrue(lo <= v <= hi,
                        f"arch_similarity={v:.3f} outside [{lo:.2f}, {hi:.2f}]: {label}")
    test.__doc__ = f"[{lo:.2f}-{hi:.2f}] {label}"
    return test


class TestArchUID(unittest.TestCase):
    """Every SAME/DIFF case must hold for the architecture certificate."""


class TestArchSimilarity(unittest.TestCase):
    """Graded measure -- reporting only, never gates identity."""


for _i, (_expect, _label, _a, _b) in enumerate(CASES):
    setattr(TestArchUID, f"test_{_i:02d}_{_expect.lower()}_{_slug(_label)}",
            _make_uid_test(_expect, _label, _a, _b))

for _i, (_label, _a, _b, _lo, _hi) in enumerate(SIM_CASES):
    setattr(TestArchSimilarity, f"test_{_i:02d}_{_slug(_label)}",
            _make_sim_test(_label, _a, _b, _lo, _hi))


if __name__ == "__main__":
    unittest.main(verbosity=2)