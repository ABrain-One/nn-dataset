"""Architecture-identity fingerprints computed from model source code.

Implements the AST def-use graph described in the Unique-ID proposal, with one
deliberate change: the graph is built from *source text* via ``ast.parse``
rather than from a live class via ``inspect.getsource``/``inspect.getmro``.

Rationale for the change:

* the dataset holds code as text, and a large share of generated candidates do
  not import at all (that is precisely why they are interesting), so any
  identity function that requires instantiating the model cannot be applied to
  them;
* importing a model to identify it executes untrusted generated code;
* ``torch.fx`` tracing has the same instantiation requirement and additionally
  only captures ``forward``, missing ``__init__``, ``train_setup`` and
  ``learn`` -- the omission the proposal's follow-up note is about.

Everything downstream (feature hashing, continuous WL embedding, discrete WL
certificate) follows the proposal. Two operating points are exposed:

``arch_uid(src)``  exact WL certificate, for grouping isomorphic duplicates
``arch_vec(src)``  unit vector, cosine = dot product, for graded similarity
"""

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from collections import defaultdict

import numpy as np

D_OP, D_HP = 256, 256  # hashing dimensions (feature-hashing / "hashing trick")
W_EFFECT = 0.5         # weight of ordering edges between side-effecting statements
UNROLL_CAP = 64        # largest literal loop that is expanded rather than summarised
NODE_BUDGET = 20000    # stop expanding once a graph is this large (nested loops)


def _hp_repr(hp: dict) -> str:
    """Canonical text of a node's parameter bag, independent of insertion order."""
    return repr(sorted((k, sorted(v)) for k, v in hp.items()))


def _bucket(s: str, d: int) -> int:
    """Stable hash -> bucket. (Python's built-in hash() is salted per process.)"""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") % d


def _scale(v: float) -> float:
    """Signed log compression, so 3 and 4096 land on a comparable scale."""
    return float(np.sign(v) * np.log1p(abs(v)))


def _const_eval(node: ast.AST, env: dict | None = None):
    """Evaluate an arithmetic expression over numeric literals.

    ``ast.literal_eval`` refuses ``BinOp``, so it cannot fold ``256 * 6 * 6`` to
    ``9216`` -- meaning the two spellings of the same flatten width would
    otherwise receive different identities. This closes that gap for the
    arithmetic that actually appears in layer definitions."""
    if isinstance(node, ast.Constant):
        v = node.value
        return None if isinstance(v, bool) or not isinstance(v, (int, float)) else v
    # A loop variable with a known value is a constant at that iteration, so
    # `32*i+32` unrolls to the same widths someone would have written out.
    if isinstance(node, ast.Name) and env and node.id in env:
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _const_eval(node.operand, env)
        return None if v is None else (v if isinstance(node.op, ast.UAdd) else -v)
    if isinstance(node, ast.BinOp):
        l, r = _const_eval(node.left, env), _const_eval(node.right, env)
        if l is None or r is None:
            return None
        try:
            for typ, fn in ((ast.Add, lambda a, b: a + b), (ast.Sub, lambda a, b: a - b),
                            (ast.Mult, lambda a, b: a * b), (ast.Div, lambda a, b: a / b),
                            (ast.FloorDiv, lambda a, b: a // b), (ast.Mod, lambda a, b: a % b),
                            (ast.Pow, lambda a, b: a ** b)):
                if isinstance(node.op, typ):
                    out = fn(l, r)
                    return out if isinstance(out, (int, float)) else None
        except (ZeroDivisionError, OverflowError, ValueError):
            return None
    return None


def _tail(node: ast.AST) -> str:
    """Final component of a dotted path: ``nn.Conv2d`` -> 'Conv2d'. The prefix is
    import/aliasing style, not semantics."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _literal_range(it: ast.AST):
    """`range(<literals>)` -> the concrete values, else None.

    A trip count is only unknowable in general; when it is written as a literal
    it is perfectly knowable, and that is the case that decides whether a stack
    of layers built by a loop matches the same stack written out."""
    if not isinstance(it, ast.Call) or _tail(it.func) != "range":
        return None
    vals = [_const_eval(a) for a in it.args]
    if not vals or any(v is None for v in vals):
        return None
    try:
        return list(range(*[int(v) for v in vals]))
    except (TypeError, ValueError):
        return None


def _name_used(name: str, body) -> bool:
    """Is a loop variable actually read? If not, materialising it would add
    nodes the written-out form does not have."""
    for stmt in body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load):
                return True
    return False


class _Graph:
    """Untyped node/edge store. A node carries an op key and a bag of scalar
    parameters; nothing else. Identifiers never become nodes."""

    def __init__(self):
        self.keys, self.hps, self.edges = [], [], []
        # node id -> ordered element ids, for values that behave as containers
        # (a list being built up, a comprehension result). Lets a layer stack
        # assembled by append be compared with one written out literally.
        self.containers: dict[int, list] = {}

    def add(self, kind: str, key: str = "") -> int:
        self.keys.append(f"{kind}|{key}")
        self.hps.append({})
        return len(self.keys) - 1

    def param(self, i: int, slot: str, val: float):
        # Stored exactly. The certificate hashes these values, so anything that
        # passes through a libm (log1p differs in its last digit between numpy
        # builds) makes the same file hash differently on two machines. Scaling
        # is applied only where the embedding is built.
        self.hps[i].setdefault(slot, []).append(float(val))

    def link(self, src, dst, port: int = 0, w: float = 1.0):
        if src is not None and dst is not None:
            self.edges.append((src, dst, port, w))


class _Dataflow(ast.NodeVisitor):
    """Turns function bodies into a def-use graph.

    Normalizations: variables are edges rather than nodes (alpha-renaming taken
    to its limit), literal subtrees are folded, and statement order is discarded
    except for statements whose result is discarded -- those are kept in source
    order because they exist for their side effects.

    Deliberately not attempted: dead-code elimination, loop unrolling, sorting of
    "unordered" constructs, and filling in callee defaults.
    """

    def __init__(self, g: _Graph, attrs: dict, uses: dict):
        self.g = g            # shared graph across all functions in the file
        self.attrs = attrs    # attribute -> producing nodes (shared)
        self.uses = uses      # attribute -> consuming nodes (shared)
        self.env = {}         # local binding -> producing node (never emitted)
        self.consts = {}      # local binding -> known numeric value
        self.effect = None    # tail of the side-effect chain
        self.outputs = []     # return values, wired to the function's def node
        self.sigs = {}        # local callee -> its parameter names, in order
        self.stores = []      # values bound to attributes, wired likewise
        self._mutated = False

    # ---- binding and lookup: the only place identifiers are used -----------
    def _bind(self, target, node):
        if isinstance(target, ast.Name):
            self.env[target.id] = node
        elif isinstance(target, ast.Attribute):
            self.attrs.setdefault(target.attr, []).append(node)
            self.stores.append(node)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for t in target.elts:
                self._bind(t, node)

    def _defined(self, node) -> bool:
        """Does the file itself define this symbol? If so it is a variable and
        must be erased; if not it is external and its name is its identity."""
        if isinstance(node, ast.Name):
            return node.id in self.env
        if isinstance(node, ast.Attribute):
            return node.attr in self.attrs or node.attr in self.uses
        return False

    def _read(self, node):
        if isinstance(node, ast.Name):
            # `env.get(...) or add("free")` silently loses node id 0, which is
            # falsy -- so the first local bound in a file would be read as an
            # unknown external symbol instead of as its own definition.
            bound = self.env.get(node.id)
            return self.g.add("free") if bound is None else bound
        if isinstance(node, ast.Attribute):
            i = self.g.add("use")
            self.uses.setdefault(node.attr, []).append(i)
            return i
        return None

    # ---- expressions -------------------------------------------------------
    def expr(self, node):
        # An empty list is a container being opened, not a constant. It must be
        # recognised before folding, which would otherwise collapse it.
        if isinstance(node, (ast.List, ast.Tuple)) and not node.elts:
            i = self.g.add("pack")
            self.g.containers[i] = []
            return i

        folded = self._fold(node)
        if folded is not None:
            return folded

        if isinstance(node, ast.Call):
            f = node.func
            mutated = self._container_mutation(node)
            if mutated is not None:
                return mutated
            # `super(Cls, self)` and `super()` are the same call written two ways,
            # so the explicit form's arguments must not reach the graph.
            if _tail(f) == "super" and node.args:
                node = ast.Call(func=f, args=[], keywords=[])
            if self._defined(f):
                i = self.g.add("call")
                self.g.link(self._read(f), i, 0)
            else:
                i = self.g.add("call", _tail(f))
                if isinstance(f, ast.Attribute) and self._defined(f.value):
                    self.g.link(self._read(f.value), i, 0)
                elif isinstance(f, ast.Attribute) and isinstance(
                        f.value, (ast.Call, ast.Subscript)):
                    # Chained call: the receiver is a computed value, as in
                    # `TorchVision('mobilenet_v3_large').to(device)`. Without this
                    # the whole receiver subtree is discarded, so two models
                    # differing only in the constructor argument collapse. A plain
                    # dotted path (`nn.Conv2d`) is deliberately NOT evaluated,
                    # because that prefix is import style rather than semantics.
                    self.g.link(self.expr(f.value), i, 0)
                elif isinstance(f, (ast.Subscript, ast.Call)):
                    self.g.link(self.expr(f), i, 0)
            port = 0
            for a in node.args:
                # `f(*layers)` must build the same graph as `f(l0, l1, l2)`, or a
                # stack assembled in a list never matches one spelled out.
                if isinstance(a, ast.Starred):
                    inner = self.expr(a.value)
                    for e in self.g.containers.get(inner, [inner]):
                        self.g.link(e, i, port)
                        port += 1
                    continue
                self._arg(i, a, port, slot=f"a{port}")
                port += 1
            # At a call to something defined in this file, a keyword is the
            # callee's own parameter name -- an identifier, so it must be erased
            # like any other. It is resolved to the parameter's position, which
            # also makes `Block(64, factor=4)` and `Block(64, 4)` the same call.
            # At a library call the keyword is part of the library's contract
            # and stays.
            sig = None
            if self._defined(f):
                sig = self.sigs.get(_tail(f))
            elif (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                  and (f.value.id == "self" or self._defined(f.value))):
                sig = self.sigs.get(_tail(f))      # self.method(x, k=1)
            for kw in node.keywords:
                # several classes may define a method of this name; the
                # keyword is resolved against whichever signature has it
                k = next((sg.index(kw.arg) for sg in (sig or []) if kw.arg in sg), None)
                if k is not None:
                    self._arg(i, kw.value, k, slot=f"a{k}")
                else:
                    self._arg(i, kw.value, port, slot=kw.arg or "kw")
            return i

        if isinstance(node, (ast.Name, ast.Attribute)):
            if self._defined(node) or isinstance(node, ast.Attribute):
                return self._read(node)
            return self.g.add("free", _tail(node) if isinstance(node, ast.Name) else "")

        if isinstance(node, ast.BinOp):
            i = self.g.add("op", type(node.op).__name__)
            self.g.link(self.expr(node.left), i, 0)
            self.g.link(self.expr(node.right), i, 1)
            return i

        if isinstance(node, ast.Subscript):
            i = self.g.add("index")
            self.g.link(self.expr(node.value), i, 0)
            # The index selects what is read, so it is part of the computation.
            # Dropping it makes `x[:, :half]` and `x[:, half:]` identical, which
            # is exactly how a channel split is written.
            sl = node.slice
            parts = ([sl.lower, sl.upper, sl.step] if isinstance(sl, ast.Slice)
                     else [sl])
            for port, part in enumerate(parts, start=1):
                if part is None:
                    self.g.param(i, f"s{port}", 0.0)   # an omitted bound is itself information
                elif isinstance(part, ast.Slice):
                    self.g.param(i, f"s{port}", 1.0)
                else:
                    self.g.link(self.expr(part), i, port)
            return i

        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            i = self.g.add("pack")
            self.g.containers[i] = [self.expr(e) for e in node.elts]
            return i

        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            # A comprehension over a literal range is a written-out sequence; it
            # is expanded so it matches the same sequence built any other way.
            gens = node.generators
            vals = _literal_range(gens[0].iter) if len(gens) == 1 else None
            if (vals is not None and not gens[0].ifs and len(vals) <= UNROLL_CAP
                    and len(self.g.keys) < NODE_BUDGET):
                i = self.g.add("pack")
                elems = []
                uses = _name_used(gens[0].target.id, [node.elt]) \
                    if isinstance(gens[0].target, ast.Name) else True
                for val in vals:
                    if uses:
                        c = self.g.add("const")
                        self.g.param(c, "v", float(val))
                        self._bind(gens[0].target, c)
                        if isinstance(gens[0].target, ast.Name):
                            self.consts[gens[0].target.id] = float(val)
                    elems.append(self.expr(node.elt))
                self.g.containers[i] = elems
                return i
            i = self.g.add("loop")
            for c in gens:
                self._bind(c.target, i)
                self.g.link(self.expr(c.iter), i, 0)
            self.g.link(self.expr(node.elt), i, 1)
            return i

        if isinstance(node, ast.Slice):
            # lower, upper and step must occupy distinct ports, or `x[:half]`
            # and `x[half:]` are indistinguishable, which is how a channel split
            # is written.
            i = self.g.add("slice")
            for port, part in enumerate((node.lower, node.upper, node.step)):
                if part is None:
                    self.g.param(i, f"n{port}", 1.0)
                else:
                    self.g.link(self.expr(part), i, port)
            return i

        if isinstance(node, ast.IfExp):
            i = self.g.add("branch")
            for k, sub in enumerate((node.test, node.body, node.orelse)):
                self.g.link(self.expr(sub), i, k)
            return i

        i = self.g.add("expr", type(node).__name__)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.g.link(self.expr(child), i, 0)
        return i

    def _container_mutation(self, node: ast.Call):
        """`layers.append(x)` / `.extend(xs)` -> record x as an element.

        Modelled as membership rather than as a call, because the written-out
        form of the same stack contains no append at all. Returns the container
        node, or None if this is an ordinary call."""
        f = node.func
        if not isinstance(f, ast.Attribute) or f.attr not in ("append", "extend"):
            return None
        if not self._defined(f.value):
            return None
        tgt = self._read(f.value)
        if tgt not in self.g.containers:
            return None
        for a in node.args:
            v = self.expr(a)
            if f.attr == "extend" and v in self.g.containers:
                self.g.containers[tgt].extend(self.g.containers[v])
            else:
                self.g.containers[tgt].append(v)
        self._mutated = True
        return tgt

    def _fold(self, node):
        """Literal subtree -> a constant node. Numbers become scalar parameters;
        strings are hashed into the op key, since a string argument selects a
        behaviour just as an enum would."""
        arith = _const_eval(node, self.consts)
        if arith is not None:
            i = self.g.add("const")
            self.g.param(i, "v", float(arith))
            return i
        try:
            val = ast.literal_eval(node)
        except Exception:
            return None
        if isinstance(val, str):
            return self.g.add("const", val)
        if isinstance(val, bool) or isinstance(val, (int, float)):
            i = self.g.add("const")
            self.g.param(i, "v", float(val))
            return i
        if isinstance(val, (tuple, list)) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in val
        ):
            i = self.g.add("const")
            for k, v in enumerate(val):
                self.g.param(i, f"v{k}", float(v))
            return i
        return None

    def _arg(self, call, node, port, slot):
        arith = _const_eval(node, self.consts)
        if arith is not None:
            self.g.param(call, slot, float(arith))
            return
        try:
            val = ast.literal_eval(node)
        except Exception:
            self.g.link(self.expr(node), call, port)
            return
        if isinstance(val, bool) or isinstance(val, (int, float)):
            self.g.param(call, slot, float(val))
        elif isinstance(val, str):
            self.g.param(call, f"{slot}={val}", 1.0)
        elif isinstance(val, (tuple, list)) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in val
        ):
            for k, v in enumerate(val):
                self.g.param(call, f"{slot}[{k}]", float(v))
        else:
            self.g.link(self.expr(node), call, port)

    # ---- statements --------------------------------------------------------
    def _store(self, target, value):
        if isinstance(target, ast.Subscript):
            self._setitem(target, value)
        else:
            self._bind(target, value)

    def _setitem(self, target: ast.Subscript, value):
        """`c[i] = v`. On a literal list with a literal index the element is
        replaced, so `[32, 64]` followed by `ws[0] = 16` matches `[16, 64]`
        written out. Anything else -- a dict, a parameter such as `prm`, a
        computed index -- becomes a `setitem` node that later reads flow through,
        so the write is part of the identity rather than silently dropped."""
        cont = target.value
        if self._defined(cont):
            c = self._read(cont)
            elems = self.g.containers.get(c)
            try:
                idx = _const_eval(target.slice, self.consts)
            except Exception:
                idx = None
            if (elems is not None and idx is not None and float(idx).is_integer()
                    and -len(elems) <= int(idx) < len(elems)):
                elems[int(idx)] = value
                return
            # A numeric list literal was folded into one const node carrying
            # v0, v1, ...; a literal write into it edits that slot in place, so
            # `[32, 64]` then `ws[0] = 16` matches `[16, 64]` written out.
            hp = self.g.hps[c]
            if (idx is not None and float(idx).is_integer()
                    and self.g.keys[c] == "const|" and self.g.keys[value] == "const|"
                    and len(self.g.hps[value].get("v", [])) == 1):
                slots = sorted(k for k in hp if re.fullmatch(r"v\d+", k))
                k = int(idx) if int(idx) >= 0 else len(slots) + int(idx)
                if 0 <= k < len(slots):
                    hp[f"v{k}"] = list(self.g.hps[value]["v"])
                    return
        else:
            c = self.expr(cont)
        i = self.g.add("setitem")
        self.g.link(c, i, 0)
        self.g.link(self.expr(target.slice), i, 1)
        self.g.link(value, i, 2)
        self._bind(cont, i)

    def visit_Assign(self, node):
        v = self.expr(node.value)
        for t in node.targets:
            self._store(t, v)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._store(node.target, self.expr(node.value))

    def visit_AugAssign(self, node):
        i = self.g.add("op", type(node.op).__name__)
        self.g.link(self.expr(node.target), i, 0)
        self.g.link(self.expr(node.value), i, 1)
        self._store(node.target, i)

    def visit_Expr(self, node):
        """The result is thrown away, so the statement exists for its effect.
        Chaining these preserves zero_grad -> backward -> step ordering without
        any knowledge of what those calls do."""
        # A bare string statement is a docstring: documentation, not computation.
        # Left in, it would hash into the identity and make an edited comment
        # look like a different architecture.
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        self._mutated = False
        i = self.expr(node.value)
        if self._mutated:
            # Filling a container is captured by element order, so it must not
            # also appear as a side effect -- the written-out form has none.
            self._mutated = False
            return
        self.g.link(self.effect, i, 0, W_EFFECT)
        self.effect = i

    def visit_Return(self, node):
        i = self.g.add("output")
        if node.value is not None:
            self.g.link(self.expr(node.value), i, 0)
        self.outputs.append(i)

    def _loop(self, node):
        # A loop over a literal range is a written-out sequence, so expand it;
        # only genuinely unknown trip counts are summarised as a loop node.
        if isinstance(node, ast.For) and not node.orelse:
            vals = _literal_range(node.iter)
            if (vals is not None and len(vals) <= UNROLL_CAP
                    and len(self.g.keys) < NODE_BUDGET):
                uses = _name_used(node.target.id, node.body) \
                    if isinstance(node.target, ast.Name) else True
                for val in vals:
                    if uses:
                        c = self.g.add("const")
                        self.g.param(c, "v", float(val))
                        self._bind(node.target, c)
                        if isinstance(node.target, ast.Name):
                            self.consts[node.target.id] = float(val)
                    for st in node.body:
                        self.visit(st)
                if isinstance(node.target, ast.Name):
                    self.consts.pop(node.target.id, None)
                return
        i = self.g.add("loop")
        if getattr(node, "iter", None) is not None:
            self.g.link(self.expr(node.iter), i, 0)
            self._bind(node.target, i)
        if getattr(node, "test", None) is not None:
            self.g.link(self.expr(node.test), i, 0)
        outer, self.effect = self.effect, i
        for s in node.body + getattr(node, "orelse", []):
            self.visit(s)
        self.g.link(self.effect, i, 1, W_EFFECT)
        self.effect = outer
        self.g.link(outer, i, 0, W_EFFECT)

    visit_For = visit_While = _loop

    def visit_If(self, node):
        i = self.g.add("branch")
        self.g.link(self.expr(node.test), i, 0)
        outer, self.effect = self.effect, i
        for s in node.body + node.orelse:
            self.visit(s)
        self.effect = outer
        self.g.link(outer, i, 0, W_EFFECT)

    def visit_With(self, node):
        i = self.g.add("ctx")
        for it in node.items:
            self.g.link(self.expr(it.context_expr), i, 0)
            if it.optional_vars is not None:
                self._bind(it.optional_vars, i)
        outer, self.effect = self.effect, i
        for s in node.body:
            self.visit(s)
        self.effect = outer
        self.g.link(outer, i, 0, W_EFFECT)


def _live_defaults(tree: ast.AST) -> dict:
    """(id(function), parameter) -> can its default value be reached?

    A default is dead when every call to that function found in the file
    supplies the argument, positionally or by keyword. Anything uncertain -- no
    call site in the file (a constructor the framework calls), a starred
    argument, a ``**kwargs`` splat -- counts as reachable. That errs towards
    keeping two files apart rather than merging them."""
    calls = defaultdict(list)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            calls[_tail(n.func)].append(n)
    out = {}
    fns = []
    for s in tree.body:
        if isinstance(s, ast.ClassDef):
            fns += [(m, s.name if m.name == "__init__" else m.name, True)
                    for m in s.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
        elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fns.append((s, s.name, False))
    for fn, callee, is_method in fns:
        a = fn.args
        pos = list(getattr(a, "posonlyargs", [])) + list(a.args)
        if is_method and pos and pos[0].arg == "self":
            pos = pos[1:]
        sites = calls.get(callee, [])
        def passed(site, name, index):
            if any(isinstance(x, ast.Starred) for x in site.args):
                return False                       # uncertain: treat as reachable
            if any(k.arg is None for k in site.keywords):
                return False
            return (index is not None and index < len(site.args)) or \
                any(k.arg == name for k in site.keywords)
        n_default = len(a.defaults)
        for k, arg in enumerate(pos[len(pos) - n_default:] if n_default else []):
            index = len(pos) - n_default + k
            out[(id(fn), arg.arg)] = not sites or not all(passed(c, arg.arg, index) for c in sites)
        for arg, d in zip(a.kwonlyargs, a.kw_defaults):
            if d is not None:
                out[(id(fn), arg.arg)] = not sites or not all(passed(c, arg.arg, None) for c in sites)
    return out


def _finalize(g: _Graph) -> _Graph:
    """Materialise containers that are still values, drop the ones that are not.

    Container elements are deliberately not wired at construction time, because
    `Sequential(*layers)` splices them straight into the call and the list that
    held them then carries nothing -- the written-out form never had it. A
    container that survives as a real value does get its elements wired here.
    Isolated constants are dropped for the same reason: a loop index that folded
    into a width is not a node the hand-written form would contain."""
    def incidence():
        seen = set()
        for s, d, _, _ in g.edges:
            seen.add(s)
            seen.add(d)
        return seen

    incident = incidence()
    for c, elems in g.containers.items():
        if c in incident:
            for port, e in enumerate(elems):
                g.link(e, c, port)
    incident = incidence()
    dead = {i for i in range(len(g.keys))
            if i not in incident and (i in g.containers or g.keys[i].startswith("const|"))}
    if not dead:
        return g
    keep = [i for i in range(len(g.keys)) if i not in dead]
    remap = {old: new for new, old in enumerate(keep)}
    out = _Graph()
    out.keys = [g.keys[i] for i in keep]
    out.hps = [g.hps[i] for i in keep]
    out.edges = [(remap[s], remap[d], p, w) for s, d, p, w in g.edges]
    out.containers = {remap[k]: [remap[e] for e in v if e in remap]
                      for k, v in g.containers.items() if k in remap}
    return out


def source_graph(src: str) -> _Graph:
    """Model source text -> one shared def-use graph over every function it
    defines (module-level helpers, methods, nested helpers alike).

    Functions are found structurally, never by name, so ``__init__``, ``forward``,
    ``train_setup``, ``learn`` and anything else the author wrote are all
    covered. Attribute reads leave placeholders that are wired to their producers
    afterwards, so nothing depends on declaration order.
    """
    tree = ast.parse(textwrap.dedent(src))
    g, attrs, uses = _Graph(), {}, {}

    # Attributes are scoped per class. A single shared namespace merges
    # `self.act` in one block with `self.act` in another, so exchanging two
    # activations between blocks only reorders one bucket and the graph is
    # unchanged. The class name is a lookup key here and is never emitted, so
    # two structurally identical classes still produce identical graphs.
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    scopes, claimed = [], set()
    for c in classes:
        members = [n for n in ast.walk(c)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if members:
            claimed.update(id(n) for n in members)
            scopes.append(members)
    loose = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and id(n) not in claimed]
    if loose:
        scopes.append(loose)
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    # Module-level statements carry design decisions too (a backbone selected at
    # import time, a module-level constant). Imports are excluded: which alias a
    # symbol arrives under is style, not semantics.
    top = [s for s in tree.body
           if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                 ast.Import, ast.ImportFrom))]
    # The module scope is built first and seeds every function's environment,
    # so a method reading a module-level constant reaches the constant rather
    # than a nameless free symbol that is later pruned as isolated.
    tables = []
    module_env, module_consts = {}, {}
    if top:
        m_attrs, m_uses = {}, {}
        tables.append((m_attrs, m_uses))
        d = _Dataflow(g, m_attrs, m_uses)
        for node in top:
            d.visit(node)
        module_env, module_consts = d.env, d.consts

    # Module-level classes and functions get a definition node, and a call to
    # one links to that node instead of carrying the name. A class node receives
    # its methods' results, so *which* helper a model uses still changes the
    # certificate; only the spelling of the helper's name stops mattering.
    defs = {}
    for s in tree.body:
        if isinstance(s, ast.ClassDef):
            c = g.add("class")
            module_env[s.name] = c
            for m in s.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs[id(m)] = g.add("def")
                    g.link(defs[id(m)], c, 0)
        elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[id(s)] = module_env[s.name] = g.add("def")

    live = _live_defaults(tree)

    def _params(fn):
        a = fn.args
        names = [x.arg for x in list(getattr(a, "posonlyargs", [])) + list(a.args)]
        return [n for n in names if n != "self"] + [x.arg for x in a.kwonlyargs]
    sigs = defaultdict(list)                  # callee name -> [signature, ...]
    for s_ in tree.body:
        if isinstance(s_, ast.ClassDef):
            for m in s_.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sigs[s_.name if m.name == "__init__" else m.name].append(_params(m))
        elif isinstance(s_, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sigs[s_.name].append(_params(s_))
    sigs = dict(sigs)

    for members in scopes:
        s_attrs, s_uses = {}, {}
        tables.append((s_attrs, s_uses))
        for node in members:
            d = _Dataflow(g, s_attrs, s_uses)
            d.env, d.consts, d.sigs = dict(module_env), dict(module_consts), sigs
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # A parameter is a variable like any other, so its spelling must
                # be erased. Binding it up front makes `forward(self, x)` and
                # `forward(self, inp)` build the same graph; left unbound it
                # would be read as an external symbol carrying its name.
                # A default that can actually be reached is part of the design
                # (`bottleneck_factor=4` vs `=8`), so it is attached to the
                # parameter; one that every call site overrides is not.
                a = node.args
                pos = list(getattr(a, "posonlyargs", [])) + list(a.args)
                pads = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
                for arg, default in list(zip(pos, pads)) + list(zip(a.kwonlyargs, a.kw_defaults)):
                    if arg.arg == "self":
                        continue
                    p = d.env[arg.arg] = g.add("param")
                    if default is not None and live.get((id(node), arg.arg), True):
                        d._arg(p, default, 0, "default")
                for extra in (a.vararg, a.kwarg):
                    if extra is not None:
                        d.env[extra.arg] = g.add("param")
                for stmt in node.body:
                    d.visit(stmt)
                dn = defs.get(id(node))
                if dn is not None:
                    for k, o in enumerate(d.outputs):
                        g.link(o, dn, k)
                    for v in d.stores:
                        g.link(v, dn, 1)
                    if d.effect is not None:
                        g.link(d.effect, dn, 0, W_EFFECT)
            else:
                d.visit(node)
    if not tables:
        _Dataflow(g, attrs, uses).visit(tree)
        tables.append((attrs, uses))

    for s_attrs, s_uses in tables:
        for name, consumers in s_uses.items():
            producers = s_attrs.get(name)
            if not producers:
                # Read but never written in this class: it is inherited, or set
                # by a sibling, so fall back to every producer in the file.
                producers = [p for o_attrs, _ in tables for p in o_attrs.get(name, [])]
            for producer in producers:
                for c in consumers:
                    g.link(producer, c, 0)
    return _finalize(g)


def featurize(g: _Graph) -> np.ndarray:
    """Node embedding: hashed op identity + hashed scalar parameters + local
    structural context. No vocabulary and no taxonomy."""
    n = len(g.keys)
    X = np.zeros((n, D_OP + D_HP + 4))
    indeg, outdeg = np.zeros(n), np.zeros(n)
    for s, d, _, _ in g.edges:
        outdeg[s] += 1
        indeg[d] += 1
    for i, key in enumerate(g.keys):
        X[i, _bucket(key, D_OP)] = 1.0
        for slot, val in g.hps[i].items():
            X[i, D_OP + _bucket(f"{key}.{slot}", D_HP)] += sum(_scale(v) for v in val)
        X[i, -4:] = [np.log1p(indeg[i]), np.log1p(outdeg[i]), 1.0, 0.0]
    return X


def adjacency(g: _Graph):
    """Two directed matrices. Direction is kept (a residual add is not its own
    reverse) and argument position is folded into the edge weight, so swapping
    operands changes the embedding."""
    n = len(g.keys)
    F = np.zeros((n, n))
    for s, d, port, w in g.edges:
        F[s, d] += w / (1.0 + port)
    return F, F.T


def wl_embed(F, B, X, iters=3, alpha=0.5) -> np.ndarray:
    """Continuous Weisfeiler-Lehman over both edge directions: hash-relabelling
    replaced by neighbourhood averaging. Permutation-invariant, fixed-size,
    deterministic, training-free."""
    dF = np.maximum(F.sum(1, keepdims=True), 1.0)
    dB = np.maximum(B.sum(1, keepdims=True), 1.0)
    H, pooled = X, [X.mean(0), X.max(0)]
    for _ in range(iters):
        H = alpha * H + (1 - alpha) * 0.5 * ((F @ H) / dF + (B @ H) / dB)
        pooled += [H.mean(0), H.max(0)]
    v = np.concatenate(pooled)
    return v / (np.linalg.norm(v) + 1e-9)


def wl_hash(g: _Graph, iters=4) -> str:
    """Exact companion to the embedding: discrete WL refinement to a canonical
    certificate, for gating literal duplicates before similarity is consulted."""
    lab = {i: hashlib.blake2b(
        (k + _hp_repr(g.hps[i])).encode(), digest_size=8).hexdigest()
        for i, k in enumerate(g.keys)}
    inc = {i: [] for i in range(len(g.keys))}
    for s, d, port, w in g.edges:
        inc[d].append((port, w, s))
    for _ in range(iters):
        lab = {i: hashlib.blake2b(
            (lab[i] + repr(sorted((p, w, lab[s]) for p, w, s in inc[i]))).encode(),
            digest_size=8).hexdigest()
            for i in range(len(g.keys))}
    return hashlib.blake2b(repr(sorted(lab.values())).encode(), digest_size=16).hexdigest()


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------

def wl_labels(g: _Graph, iters=3) -> set:
    """Multiset of Weisfeiler-Lehman subtree labels, pooled over refinement rounds.

    The exact certificate collapses these into one hash, which answers only
    'identical or not'. Keeping the labels themselves supports a graded answer:
    two architectures sharing most of their subtrees share most of their labels.
    This is the classical WL subtree kernel, and unlike a pooled continuous
    embedding it does not wash out when many models share a common skeleton --
    shared structure contributes shared labels rather than pulling every vector
    toward the same mean."""
    lab = {i: hashlib.blake2b(
        (k + _hp_repr(g.hps[i])).encode(), digest_size=8).hexdigest()
        for i, k in enumerate(g.keys)}
    inc = {i: [] for i in range(len(g.keys))}
    for s, d, port, w in g.edges:
        inc[d].append((port, w, s))
    bag = {f"0:{v}" for v in lab.values()}
    for it in range(iters):
        lab = {i: hashlib.blake2b(
            (lab[i] + repr(sorted((p, w, lab[s]) for p, w, s in inc[i]))).encode(),
            digest_size=8).hexdigest()
            for i in range(len(g.keys))}
        bag |= {f"{it + 1}:{v}" for v in lab.values()}
    return bag


def arch_similarity(a: str, b: str, iters=3) -> float:
    """Graded architectural similarity in [0, 1] -- Jaccard over WL subtree labels.

    1.0 means the certificates agree; lower means genuinely less shared structure.
    Use `arch_uid` for identity and this only for reporting or ranking, never as a
    threshold that decides row identity."""
    la, lb = wl_labels(source_graph(a), iters), wl_labels(source_graph(b), iters)
    if not la and not lb:
        return 1.0
    return len(la & lb) / len(la | lb)


def arch_uid(src: str) -> str:
    """Exact architectural certificate for a model given as source text."""
    return wl_hash(source_graph(src))


def arch_vec(src: str) -> np.ndarray:
    """Unit-norm architectural embedding; cosine similarity is a dot product."""
    g = source_graph(src)
    return wl_embed(*adjacency(g), featurize(g))


def arch_uid_and_vec(src: str):
    """Both operating points from a single graph construction."""
    g = source_graph(src)
    return wl_hash(g), wl_embed(*adjacency(g), featurize(g)), len(g.keys)


# --------------------------------------------------------------------------
# scope selection: which region of the file is the model's own design
# --------------------------------------------------------------------------

_HEAD_OPEN = re.compile(r"^#\s*=+\s*LLM-GENERATED HEAD\s*=+\s*$", re.M)
_HEAD_CLOSE = re.compile(r"^#\s*=+\s*$", re.M)
_BACKBONE = re.compile(r"_edge_backbone\(\s*['\"]([A-Za-z0-9_]+)['\"]")


def has_scaffold(src: str) -> bool:
    """True for template-arm models, which carry a fixed shared scaffold."""
    return _HEAD_OPEN.search(src) is not None


def authored_region(src: str) -> str:
    """The part of the file its author actually designed.

    For template-arm models this is the head block between the scaffold seam
    markers, plus the backbone selection, plus the hyperparameter declaration --
    exactly the three things the generator writes. For every other model (free
    form, and all human-written LEMUR entries) there is no scaffold, so the
    authored region is the whole file.
    """
    m = _HEAD_OPEN.search(src)
    if not m:
        return src
    rest = src[m.end():]
    close = _HEAD_CLOSE.search(rest)
    head = rest[:close.start()] if close else rest

    extra = []
    bb = _BACKBONE.search(src)
    if bb:
        # the backbone choice is a design decision and must count toward identity
        extra.append(f"_backbone_choice('{bb.group(1)}')")
    hp = re.search(r"def supported_hyperparameters\(\):.*?\n\n", src, re.S)
    if hp:
        extra.append(hp.group(0))
    return head + "\n" + "\n".join(extra)
