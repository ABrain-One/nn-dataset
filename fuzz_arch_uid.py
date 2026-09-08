"""Differential fuzzing for the architecture identity.

Rather than wait for counterexamples to be reported one at a time, this generates
them. Every real model in a corpus is mutated two ways:

* **invariant** rewrites, which change the text but not the network. The identity
  must not move. A failure here means the identity is too strict and real
  duplicates will be missed.
* **semantic** rewrites, which change the network. The identity must move. A
  failure here means the identity is too loose and different models will be
  merged, which is the class of defect that matters for a dataset key.

The mutations are AST transforms, so they apply to arbitrary model source rather
than to hand-written examples, and the swap mutation is deliberately the one that
exposes shared-namespace collisions between sibling classes.

    python fuzz_arch_uid.py --n 400                 # this checkout's ab/nn/nn
    python fuzz_arch_uid.py --dir /other/models --n 400 --show 3
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import random
from collections import Counter
from pathlib import Path

from ab.nn.util.ArchUID import arch_uid
from ab.nn.util.Const import nn_dir



# ----------------------------------------------------------------- mutations
class _RenameAttrs(ast.NodeTransformer):
    """Rename attributes reached through `self` only.

    Renaming every matching attribute would also rewrite external calls such as
    `F.relu`, which is a semantic change, and the resulting failures would be
    the fuzzer's fault rather than the identity's."""

    def __init__(self, mapping):
        self.m = mapping

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if (node.attr in self.m and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            node.attr = self.m[node.attr]
        return node


class _RenameLocals(ast.NodeTransformer):
    def __init__(self, mapping, local_callees=()):
        self.m = mapping
        self.local = set(local_callees)

    def visit_Name(self, node):
        if node.id in self.m:
            node.id = self.m[node.id]
        return node

    def visit_arg(self, node):
        if node.arg in self.m:
            node.arg = self.m[node.arg]
        return node

    def visit_Call(self, node):
        # A parameter passed by keyword must be renamed at the call site too, or
        # the mutated program is not a rename of the original: it no longer
        # passes that argument, so the callee's default becomes reachable. Only
        # calls to callables defined in this file are touched -- keywords of
        # library calls belong to the library's signature, not to ours.
        self.generic_visit(node)
        f = node.func
        tail = f.attr if isinstance(f, ast.Attribute) else (
            f.id if isinstance(f, ast.Name) else None)
        if tail in self.local:
            for kw in node.keywords:
                if kw.arg in self.m:
                    kw.arg = self.m[kw.arg]
        return node


class _AddDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.body.insert(0, ast.Expr(value=ast.Constant(value="Documentation.")))
        return node


class _Annotate(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        for a in node.args.args:
            if a.arg != "self" and a.annotation is None:
                a.annotation = ast.Name(id="object", ctx=ast.Load())
        return node


def _call_names(tree):
    """Constructor calls, as (node, dotted-tail) pairs."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            tail = f.attr if isinstance(f, ast.Attribute) else (
                f.id if isinstance(f, ast.Name) else None)
            if tail and tail[:1].isupper():
                out.append((n, tail))
    return out


def mut_rename_attrs(tree, rnd):
    """Rename only attributes reached exclusively through `self`.

    If the same attribute name is also read off another object, renaming the
    `self` occurrences alone changes which attribute the program refers to, so
    the rewrite would not be an alpha-rename and any resulting mismatch would be
    the fuzzer's fault."""
    assigned = {t.attr for n in ast.walk(tree) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name) and t.value.id == "self"}
    via_other = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                 and not (isinstance(n.value, ast.Name) and n.value.id == "self")}
    names = sorted(assigned - via_other)
    if not names:
        return None
    return _RenameAttrs({n: f"zz{i}" for i, n in enumerate(names)}).visit(tree)


def mut_rename_locals(tree, rnd):
    names = sorted({t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)}
                   | {a.arg for f in ast.walk(tree)
                      if isinstance(f, ast.FunctionDef) for a in f.args.args
                      if a.arg != "self"})
    if not names:
        return None
    local = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
    return _RenameLocals({n: f"vv{i}" for i, n in enumerate(names)}, local).visit(tree)


def mut_docstrings(tree, rnd):
    return _AddDocstrings().visit(tree)


def mut_annotate(tree, rnd):
    return _Annotate().visit(tree)


def mut_swap_two_ops(tree, rnd):
    """Exchange two DIFFERENT constructors between their positions.

    This is the mutation that finds shared-namespace collisions: if two sibling
    classes both store their result under the same attribute name, exchanging
    them leaves a single merged bucket unchanged."""
    calls = _call_names(tree)
    by_name = {}
    for n, t in calls:
        by_name.setdefault(t, []).append(n)
    distinct = [t for t in by_name if len(by_name[t]) >= 1]
    if len(distinct) < 2:
        return None
    a, b = rnd.sample(distinct, 2)
    na, nb = by_name[a][0], by_name[b][0]
    na.func, nb.func = nb.func, na.func
    na.args, nb.args = nb.args, na.args
    na.keywords, nb.keywords = nb.keywords, na.keywords
    return tree


def mut_change_number(tree, rnd):
    consts = [n for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and isinstance(n.value, int)
              and not isinstance(n.value, bool) and n.value not in (0, 1)]
    if not consts:
        return None
    c = rnd.choice(consts)
    c.value = c.value * 2 + 7
    return tree


def mut_drop_stmt(tree, rnd):
    """Remove one attribute assignment, i.e. delete a layer."""
    holders = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and sum(1 for s in n.body if isinstance(s, ast.Assign)
                       and any(isinstance(t, ast.Attribute) for t in s.targets)) >= 2]
    if not holders:
        return None
    f = rnd.choice(holders)
    idx = [i for i, s in enumerate(f.body) if isinstance(s, ast.Assign)
           and any(isinstance(t, ast.Attribute) for t in s.targets)]
    del f.body[rnd.choice(idx)]
    return tree


INVARIANT = [("rename attributes", mut_rename_attrs),
             ("rename locals and parameters", mut_rename_locals),
             ("add docstrings", mut_docstrings),
             ("add type annotations", mut_annotate)]
SEMANTIC = [("swap two constructors", mut_swap_two_ops),
            ("change a numeric literal", mut_change_number),
            ("delete a layer assignment", mut_drop_stmt)]


# ----------------------------------------------------------------- driver
def corpus(where, n, rnd):
    fs = sorted(glob.glob(str(Path(where) / "**" / "*.py"), recursive=True))
    fs = [f for f in fs if not f.endswith("__init__.py")]
    rnd.shuffle(fs)
    out = []
    for f in fs:
        if len(out) >= n:
            break
        try:
            src = Path(f).read_text(errors="replace")
            ast.parse(src)
        except Exception:
            continue
        out.append((Path(f).stem, src))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(nn_dir), help="directory of model .py files")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--show", type=int, default=2, help="example failures to print")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rnd = random.Random(a.seed)
    models = corpus(a.dir, a.n, rnd)
    print(f"fuzzing {len(models)} models from {a.dir}\n")

    ran = Counter()
    failed = Counter()
    examples = {}
    for name, src in models:
        try:
            base = arch_uid(src)
        except Exception:
            continue
        for label, fn, must_change in ([(l, f, False) for l, f in INVARIANT]
                                       + [(l, f, True) for l, f in SEMANTIC]):
            try:
                tree = fn(ast.parse(src), rnd)
                if tree is None:
                    continue
                ast.fix_missing_locations(tree)
                mutated = ast.unparse(tree)
                if mutated == ast.unparse(ast.parse(src)):
                    continue                       # mutation was a no-op
                uid = arch_uid(mutated)
            except Exception:
                continue
            ran[label] += 1
            changed = (uid != base)
            if changed != must_change:
                failed[label] += 1
                examples.setdefault(label, (name, src, mutated))

    print(f"{'mutation':<34}{'kind':<12}{'ran':>7}{'failed':>8}{'rate':>8}")
    print("-" * 70)
    for label, _ in INVARIANT:
        r, f = ran[label], failed[label]
        if r:
            print(f"{label:<34}{'invariant':<12}{r:>7}{f:>8}{f/r:>8.1%}")
    for label, _ in SEMANTIC:
        r, f = ran[label], failed[label]
        if r:
            print(f"{label:<34}{'semantic':<12}{r:>7}{f:>8}{f/r:>8.1%}")
    print("-" * 70)
    tot_r, tot_f = sum(ran.values()), sum(failed.values())
    print(f"{'total':<46}{tot_r:>7}{tot_f:>8}{tot_f/max(tot_r,1):>8.1%}")
    if not tot_f:
        print("\nno counterexamples found")
        return
    print(f"\n{'=' * 70}\nEXAMPLE FAILURES\n{'=' * 70}")
    import difflib
    for label, (name, src, mut) in list(examples.items())[:a.show]:
        print(f"\n--- {label}   model {name[:44]}")
        d = list(difflib.unified_diff(ast.unparse(ast.parse(src)).splitlines(),
                                      mut.splitlines(), lineterm="", n=0))[2:]
        for line in d[:12]:
            print("    " + line[:104])


if __name__ == "__main__":
    main()
