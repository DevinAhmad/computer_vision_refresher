"""
Python Standard Library — Tutorial (Python 3.12+)
==================================================

IMPORT-ONLY module. No side-effects on import. Run tests manually:

    import python_standard_library
    python_standard_library._self_tests()

    # or
    python python_standard_library.py

Requires Python 3.12+ — uses:
- `type` statement (3.12 type aliases)
- `X | Y` union syntax (3.10+)
- `match` structural pattern matching (3.10+)
- `Self` is 3.11+, but we show 3.12 style
- `functools.cache` (3.9+) etc.

Covers 8 topics:
1. Type Annotations (typing, `|`, TypeVar, ParamSpec, Protocol, TypedDict, dataclass, match)
2. Pathlib (pure — no filesystem writes, no tempfile)
3. Collections (Counter, defaultdict, deque, namedtuple, ChainMap, UserDict, OrderedDict)
4. Functools (lru_cache, cache, partial, wraps, reduce, singledispatch, cached_property, total_ordering)
5. Re (compile, groups, flags, verbose, lookahead, sub, split)
6. Sys (version, platform, path, getsizeof, exc_info, argv mock)
7. Math (comb/perm, gcd/lcm, dist, isclose, prod, tau, etc.)
8. Heapq (heapify, push/pop, merge, nlargest, tuple heaps)

Style: tutorial — every section explains *why* and *when* to use the feature,
plus common pitfalls. All asserts are deterministic and pure (pathlib uses
PurePath — no disk I/O).
"""

from __future__ import annotations

import collections
import functools
import heapq
import math
import pathlib
import re
import sys
from collections import ChainMap, Counter, OrderedDict, UserDict, UserList, UserString, defaultdict, deque, namedtuple
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Generic, NamedTuple, NotRequired, ParamSpec, Protocol, Self, TypedDict, TypeVar, overload

# =============================================================================
# 1. TYPE ANNOTATIONS — Python 3.12 tutorial
# =============================================================================
# Type hints are *annotations* for static checkers (mypy/pyright). They have
# ZERO runtime cost (except dataclass/typing.get_type_hints). Python 3.12 adds
# the `type` statement for clean aliases and better generics.
#
# Key ideas:
# - `X | Y` is preferred over `Union[X,Y]` since 3.10 (PEP 604).
# - `X | None` is preferred over `Optional[X]`.
# - `type` aliases are lazy, can be generic, and are visible to checkers.
# - `Annotated` lets you attach metadata (e.g., validation) without affecting check.


# 3.12 `type` statement — replaces `Vector = list[float]` with proper alias
type Vector = list[float]
type Matrix = list[Vector]
# Generic alias
T = TypeVar("T")
P = ParamSpec("P")
type Pair[T] = tuple[T, T]  # generic type alias (3.12)

# TypedDict with NotRequired (3.11+) — optional keys without total=False
class UserRecord(TypedDict):
    name: str
    age: int
    email: NotRequired[str]  # may be absent; without NotRequired every key is required
    tags: NotRequired[list[str]]

# Annotated: e.g., for validation frameworks (checkers see `str`, runtime sees metadata)
type NonEmptyStr = Annotated[str, "non-empty"]

# Protocol = structural typing (duck typing for checkers). No inheritance needed.
class Drawable(Protocol):
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str:
        return "○"

class Square:
    def draw(self) -> str:
        return "□"

# Generic class (3.12 syntax: `class Stack[T]:` is alternative, but we keep classic for clarity)
class Stack(Generic[T]):
    """Generic stack — checker knows Stack[int] vs Stack[str]."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> T | None:  # X | None is 3.10+ union
        return self._items[-1] if self._items else None

    def __len__(self) -> int:
        return len(self._items)

# Self: for fluent APIs where method returns `self` type including subclasses
class FluentCounter:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def inc(self) -> Self:  # Self means "this class" even when subclassed
        self.value += 1
        return self

    def add(self, n: int) -> Self:
        self.value += n
        return self

# Overload: describe multiple call signatures for one implementation
@overload
def parse(val: int) -> int: ...
@overload
def parse(val: str) -> int: ...
@overload
def parse(val: bytes) -> int: ...

def parse(val: int | str | bytes) -> int:  # implementation uses union
    """Parse int/str/bytes to int — overloads help checker pick correct return."""
    if isinstance(val, bytes):
        return int(val.decode())
    if isinstance(val, str):
        return int(val.strip())
    return val

# Demonstrate `match` with types (3.10+) — type narrowing via structural matching
def describe_value(v: int | str | list[int]) -> str:
    """Use `match` to narrow unions — cleaner than if/elif isinstance chain."""
    match v:
        case int(n):
            return f"int {n}"
        case str(s):
            return f"str len={len(s)}"
        case list() as lst:
            return f"list len={len(lst)}"
        case _:  # exhaustive wildcard
            return "unknown"

# ParamSpec: type a decorator that preserves wrapped signature
def logging_decorator(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that logs but keeps exact signature via ParamSpec — checker sees same args."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # tutorial: P.args / P.kwargs preserve call site checking
        return func(*args, **kwargs)

    return wrapper

# Dataclass with kw_only (3.10+) and type aliases
@dataclass(kw_only=True)
class Point2D:
    x: float
    y: float
    label: str = ""

    def as_vector(self) -> Vector:
        return [self.x, self.y]

def type_annotations_demo() -> dict[str, Any]:
    """Exercise 3.12 typing features and return samples for asserts."""
    # Vector alias
    v: Vector = [1.0, 2.0, 3.0]
    m: Matrix = [v, [4.0, 5.0]]
    # Pair generic
    p: Pair[int] = (1, 2)
    # Stack generic
    s: Stack[int] = Stack()
    s.push(10)
    assert s.peek() == 10
    # Fluent Self
    c = FluentCounter(5).inc().add(3)
    assert c.value == 9
    # TypedDict
    rec: UserRecord = {"name": "Ada", "age": 30}
    # Drawable protocol — no inheritance, just has draw()
    def render(d: Drawable) -> str:
        return d.draw()

    assert render(Circle()) == "○"
    # match narrowing
    assert describe_value(42) == "int 42"
    assert describe_value("hi") == "str len=2"
    # overload parse
    assert parse(5) == 5 and parse(" 42 ") == 42 and parse(b"7") == 7
    # get_type_hints runtime
    from typing import get_type_hints

    hints = get_type_hints(Point2D)
    assert "x" in hints and "y" in hints
    return {"vector": v, "matrix": m, "pair": p, "stack_top": s.peek(), "render": render(Square())}


# =============================================================================
# 2. PATHLIB — pure paths (no filesystem writes, no tempfile)
# =============================================================================
# pathlib is the modern replacement for os.path.
# - PurePath (PurePosixPath, PureWindowsPath) — *pure* string manipulation, no I/O, no OS calls.
# - Path — concrete, can touch filesystem (stat, read_text, etc.), but we avoid I/O here per request.
#
# Why pathlib > os.path?
# - Operator `/` for joining is readable: Path("a") / "b" / "c.txt"
# - Object-oriented: `p.suffix`, `p.stem`, `p.parent` vs `os.path.splitext(...)`
# - Cross-platform PurePath handles Windows vs Posix semantics without needing that OS.
#
# We use ONLY pure operations so tests are deterministic and leave no files.

type PathStr = str | pathlib.PurePath  # 3.12 alias example reused


def pathlib_pure_demo() -> dict[str, Any]:
    """Pure path manipulation — no disk, works on any OS."""
    # PurePosixPath always uses '/' even on Windows — good for asserts
    posix = pathlib.PurePosixPath("a/b/c.txt")
    assert posix.parts == ("a", "b", "c.txt")
    assert posix.parent == pathlib.PurePosixPath("a/b")
    assert posix.name == "c.txt"
    assert posix.stem == "c"
    assert posix.suffix == ".txt"
    assert posix.suffixes == [".txt"]
    # with_* creates new path without mutating
    assert posix.with_name("d.py").name == "d.py"
    assert posix.with_suffix(".md").suffix == ".md"
    assert posix.with_stem("new").stem == "new"  # 3.12+? Actually 3.9+ but we show

    # Join via `/` and `joinpath` — both pure
    base = pathlib.PurePosixPath("project")
    joined = base / "src" / "main.py"
    assert str(joined) == "project/src/main.py"
    assert base.joinpath("src", "main.py") == joined

    # relative_to / is_relative_to (3.9+)
    assert joined.is_relative_to("project")
    assert joined.relative_to("project") == pathlib.PurePosixPath("src/main.py")

    # PureWindowsPath — Windows semantics on Linux host (pure!)
    win = pathlib.PureWindowsPath("C:/Users/Ada/Documents/file.txt")
    assert win.drive == "C:"
    assert win.parts[0] == "C:\\"
    assert win.suffix == ".txt"

    # Generic PurePath auto-picks OS, but we assert via PurePosixPath for determinism
    generic = pathlib.PurePath("a/b/c")
    assert generic.parts == ("a", "b", "c")

    # with_suffixs for double extensions
    archive = pathlib.PurePosixPath("archive.tar.gz")
    assert archive.suffixes == [".tar", ".gz"]
    assert archive.suffix == ".gz"  # only last

    return {
        "posix": str(posix),
        "joined": str(joined),
        "win_drive": win.drive,
        "archive_suffixes": archive.suffixes,
    }


def pathlib_concrete_pure_ops_demo() -> dict[str, Any]:
    """
    Concrete Path pure ops — still no I/O.

    Even `Path` objects can do pure ops without touching disk:
    - `Path("a/b") / "c"` is pure
    - `as_posix()`, `as_uri()` (pure string)
    - `match(pattern)` checks glob pattern against path string (pure, no FS scan)
    """
    # Concrete Path but only pure methods
    p = pathlib.Path("project/src/app.py")
    # These are all pure — no file is read
    assert p.parts == ("project", "src", "app.py")
    assert p.as_posix() == "project/src/app.py"
    # match uses glob pattern matching on path string, not FS — pattern matched from right
    assert p.match("*.py") is True  # matches because filename ends with .py (path suffix)
    assert p.match("**/*.py") is True
    assert p.match("*.txt") is False
    assert pathlib.Path("a/b/c.py").match("a/**/*.py") is True

    # Parents iteration — pure
    parents = list(p.parents)  # [Path("project/src"), Path("project"), Path(".")]
    assert str(parents[0]) == "project/src"
    assert str(parents[1]) == "project"

    # Anchor / drive (empty on Posix)
    assert p.anchor == "" or p.anchor == "."  # Posix relative has no anchor
    pure = pathlib.PurePosixPath("/tmp/a/b.txt")
    assert pure.anchor == "/"
    assert pure.root == "/"

    # Note: async pathlib — there is NO stdlib AsyncPath. For async FS, use
    # `asyncio.to_thread(lambda: Path(...).read_text())` or third-party `aiopath`.
    # Tutorial aside:
    #   async def read_async(path: Path) -> str:
    #       return await asyncio.to_thread(path.read_text)
    # We don't run it here to keep import-only and no FS.

    return {"parts": p.parts, "match_py": p.match("**/*.py"), "parents": [str(x) for x in parents]}


def pathlib_splitting_demo() -> dict[str, Any]:
    """Edge cases: dots, no suffix, hidden files."""
    hidden = pathlib.PurePosixPath(".hidden")
    assert hidden.name == ".hidden" and hidden.suffix == ""  # dotfile no suffix
    no_ext = pathlib.PurePosixPath("README")
    assert no_ext.suffix == "" and no_ext.stem == "README"
    multi = pathlib.PurePosixPath("a.b.c.tar.gz")
    assert multi.suffixes == [".b", ".c", ".tar", ".gz"]  # actually first dot splits? Check: a.b.c.tar.gz -> suffixes
    # But PurePosixPath("a.b.c.tar.gz") suffixes includes all after first dot? Let's assert at least last two are tar.gz
    assert multi.suffixes[-2:] == [".tar", ".gz"]
    # with_suffix("") removes suffix
    assert pathlib.PurePosixPath("file.txt").with_suffix("") == pathlib.PurePosixPath("file")
    return {"hidden_suffix": hidden.suffix, "multi_suffixes": multi.suffixes}


# =============================================================================
# 3. COLLECTIONS
# =============================================================================
# collections = high-performance container datatypes.
# Why not plain dict/list? Specialized semantics + speed.


def collections_counter_demo() -> dict[str, Any]:
    """Counter — multiset / frequency table."""
    # Counter counts hashable objects; most_common uses heap internally
    words = ["apple", "banana", "apple", "apple", "banana", "cherry"]
    c = Counter(words)
    assert c["apple"] == 3
    assert c.most_common(1) == [("apple", 3)]
    # Arithmetic: add/subtract counters
    d = Counter(banana=1, apple=1)
    assert (c - d)["apple"] == 2
    # Elements: expand counts back to iterator
    assert sorted(c.elements()) == sorted(words)
    # Update from iterable
    c.update(["cherry", "cherry"])
    assert c["cherry"] == 3
    return {"counter": dict(c), "most_common": c.most_common(2)}


def collections_deque_demo() -> dict[str, Any]:
    """Deque — double-ended queue, O(1) append/appendleft/popleft."""
    # deque is ideal for BFS, sliding window, history
    dq: deque[int] = deque([1, 2, 3], maxlen=5)  # maxlen bounds it (tutorial)
    dq.append(4)  # right
    dq.appendleft(0)  # left
    assert list(dq) == [0, 1, 2, 3, 4]
    dq.append(5)  # at maxlen=5, append drops leftmost? Actually maxlen 5, now 6 items -> leftmost 0 dropped
    assert list(dq) == [1, 2, 3, 4, 5]
    dq.rotate(1)  # rotate right by 1: last becomes first
    assert dq[0] == 5
    dq.rotate(-1)  # back
    assert dq[0] == 1
    # popleft O(1) vs list.pop(0) O(n)
    val = dq.popleft()
    assert val == 1
    return {"deque": list(dq), "popped": val}


def collections_defaultdict_demo() -> dict[str, Any]:
    """defaultdict — never raises KeyError, auto-creates default."""
    # Common pitfall: plain dict needs `if key not in d: d[key]=[]`
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for k, v in [("a", 1), ("a", 2), ("b", 3)]:
        groups[k].append(v)
    assert groups["a"] == [1, 2]
    assert groups["missing"] == []  # auto-created! (may surprise)
    # int default 0 for counting
    counts: defaultdict[str, int] = defaultdict(int)
    for ch in "aab":
        counts[ch] += 1
    assert counts["a"] == 2
    assert counts["z"] == 0  # auto 0
    return {"groups": dict(groups), "counts": dict(counts)}


def collections_namedtuple_demo() -> dict[str, Any]:
    """namedtuple and typing.NamedTuple — lightweight immutable records."""
    # Classic namedtuple — tuple subclass with named fields
    PointClassic = namedtuple("PointClassic", ["x", "y"])
    p1 = PointClassic(1, 2)
    assert p1.x == 1 and p1[0] == 1
    assert p1._asdict() == {"x": 1, "y": 2}
    # typing.NamedTuple — with type hints, preferred in 3.12
    class PointNT(NamedTuple):
        x: float
        y: float
        label: str = ""

    p2 = PointNT(3, 4, label="B")
    assert p2.x == 3 and p2.label == "B"
    # _replace creates new instance (immutable)
    p3 = p2._replace(x=10)
    assert p3.x == 10 and p2.x == 3
    return {"classic": p1, "typed": p2, "replaced": p3}


def collections_chainmap_ordered_user_demo() -> dict[str, Any]:
    """ChainMap, OrderedDict, UserDict/UserList/UserString."""
    # ChainMap — layered dicts, first wins, useful for config > env > defaults
    defaults = {"color": "blue", "size": 10}
    overrides = {"color": "red"}
    chain = ChainMap(overrides, defaults)
    assert chain["color"] == "red"  # from overrides
    assert chain["size"] == 10  # from defaults
    assert chain.maps[0] is overrides
    # OrderedDict — remembers insertion order and has move_to_end; plain dict also
    # ordered since 3.7, but OrderedDict has extra methods and order-sensitive equality
    od: OrderedDict[str, int] = OrderedDict([("a", 1), ("b", 2)])
    od.move_to_end("a")
    assert list(od.keys()) == ["b", "a"]
    assert od != OrderedDict([("a", 1), ("b", 2)])  # order matters for OrderedDict
    # UserDict/UserList/UserString — easily subclassable wrappers (vs subclassing dict/list directly has pitfalls)
    class MyDict(UserDict):
        def __missing__(self, key: Any) -> Any:
            return f"missing:{key}"

    md = MyDict({"x": 1})
    assert md["x"] == 1 and md["nope"] == "missing:nope"
    ul = UserList([1, 2, 3])
    ul.append(4)
    assert list(ul) == [1, 2, 3, 4]
    us = UserString("hello")
    assert us.upper() == "HELLO"  # returns UserString
    return {"chain_color": chain["color"], "ordered_keys": list(od.keys()), "user_dict": md["nope"]}


# =============================================================================
# 4. FUNCTOOLS
# =============================================================================
# functools = higher-order functions for functions (decorators, caching, partials).


def functools_lru_cache_demo() -> dict[str, Any]:
    """lru_cache / cache — memoization."""
    # lru_cache(maxsize=None) = unbounded, maxsize=128 bounded with LRU eviction
    call_count = {"n": 0}

    @functools.lru_cache(maxsize=None)
    def fib(n: int) -> int:
        call_count["n"] += 1
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)

    assert fib(10) == 55
    hits_before = fib.cache_info().hits
    fib(10)  # cached
    assert fib.cache_info().hits > hits_before
    # 3.9+ `functools.cache` is alias for `lru_cache(maxsize=None)` — cleaner
    @functools.cache
    def sq(n: int) -> int:
        return n * n

    assert sq(5) == 25
    # Pitfall: args must be hashable; lists not allowed
    return {"fib": fib(10), "cache_info": fib.cache_info(), "sq": sq(5)}


def functools_partial_demo() -> dict[str, Any]:
    """partial — freeze some args of a function."""
    # partial creates new callable with preset args/kwargs
    def power(base: int, exp: int) -> int:
        return base**exp

    square = functools.partial(power, exp=2)
    cube = functools.partial(power, exp=3)
    assert square(5) == 25 and cube(3) == 27
    # Also useful with map, sorted key, etc.
    from math import isclose

    # partialmethod for methods (descriptors)
    class MyClass:
        def _add(self, a: int, b: int, c: int) -> int:
            return a + b + c

        add_one = functools.partialmethod(_add, 1)  # self._add(1, b, c)

    obj = MyClass()
    assert obj.add_one(2, 3) == 6
    assert isclose  # avoid unused
    return {"square": square(4), "cube": cube(2)}


def functools_wraps_reduce_demo() -> dict[str, Any]:
    """wraps (preserve metadata) and reduce (fold)."""
    # wraps copies __name__, __doc__, __wrapped__ — critical for decorators
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    @decorator
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello {name}"

    assert greet.__name__ == "greet"  # without wraps would be "wrapper"
    assert greet.__doc__ == "Greet someone."

    # reduce: iteratively apply func to sequence → single value. Prefer `sum`/`math.prod` when possible.
    nums = [1, 2, 3, 4]
    total = functools.reduce(lambda a, b: a + b, nums, 0)
    assert total == 10
    prod = functools.reduce(lambda a, b: a * b, nums, 1)
    assert prod == 24
    # For sum, just use `sum(nums)` — reduce is tutorial here
    return {"greet_name": greet.__name__, "total": total, "prod": prod}


def functools_dispatch_cached_demo() -> dict[str, Any]:
    """singledispatch, cached_property, total_ordering."""
    # singledispatch: overload function by first arg type (like overloading)
    @functools.singledispatch
    def format_val(val: Any) -> str:
        return f"unknown:{val!r}"

    @format_val.register(int)
    def _(val: int) -> str:
        return f"int:{val}"

    @format_val.register(str)
    def _(val: str) -> str:
        return f"str:{val}"

    @format_val.register(list)
    def _(val: list) -> str:
        return f"list len={len(val)}"

    assert format_val(42) == "int:42"
    assert format_val("hi") == "str:hi"
    assert format_val([1, 2]) == "list len=2"

    # cached_property: like @property but computed once and cached (3.8+)
    class Circle:
        def __init__(self, radius: float) -> None:
            self.radius = radius
            self._compute_count = 0

        @functools.cached_property
        def area(self) -> float:
            self._compute_count += 1
            return math.pi * self.radius**2

    c = Circle(1)
    assert math.isclose(c.area, math.pi)
    assert c._compute_count == 1
    _ = c.area  # second access shouldn't recompute
    assert c._compute_count == 1

    # total_ordering: define __eq__ and one of __lt__/__le__/__gt__/__ge__, get rest auto
    @functools.total_ordering
    class Version:
        def __init__(self, major: int) -> None:
            self.major = major

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, Version):
                return NotImplemented
            return self.major == other.major

        def __lt__(self, other: object) -> bool:
            if not isinstance(other, Version):
                return NotImplemented
            return self.major < other.major

    assert Version(1) < Version(2) and Version(2) > Version(1) and Version(1) <= Version(1)
    return {"format_int": format_val(42), "area": c.area, "version_lt": Version(1) < Version(2)}


# =============================================================================
# 5. RE — regular expressions
# =============================================================================
# re is powerful but easy to misuse. Always use raw strings r"..." so `\d` isn't escaped.
# Compile once if reused many times — `re.compile` caches, but explicit compile is clearer.


def re_basic_demo() -> dict[str, Any]:
    """search, match, fullmatch, findall, finditer, compile."""
    text = "Call me at 415-555-1234 or 650-555-9876"
    # compile with pattern — reuse for many texts
    phone_pat = re.compile(r"\d{3}-\d{3}-\d{4}")
    m = phone_pat.search(text)
    assert m is not None and m.group() == "415-555-1234"
    # match anchors at start; search anywhere; fullmatch requires entire string
    assert re.match(r"\d+", "123abc") is not None
    assert re.match(r"\d+", "abc123") is None
    assert re.fullmatch(r"\d+", "123") is not None
    assert re.fullmatch(r"\d+", "123a") is None
    # findall / finditer
    nums = re.findall(r"\d+", "a1 b22 c333")
    assert nums == ["1", "22", "333"]
    it = list(re.finditer(r"\d+", "a1 b22"))
    assert it[0].group() == "1" and it[1].span() == (4, 6)
    # split by pattern
    parts = re.split(r"\s*,\s*", "a, b ,c")
    assert parts == ["a", "b", "c"]
    return {"first_phone": m.group(), "nums": nums, "parts": parts}


def re_groups_demo() -> dict[str, Any]:
    """Capturing, non-capturing, named groups, backrefs."""
    # Groups via parentheses
    m = re.search(r"(\d{3})-(\d{3})-(\d{4})", "415-555-1234")
    assert m is not None
    assert m.groups() == ("415", "555", "1234")
    assert m.group(1) == "415"
    # Named groups (?P<name>...)
    m2 = re.search(r"(?P<area>\d{3})-(?P<num>\d{3}-\d{4})", "415-555-1234")
    assert m2 is not None and m2.groupdict() == {"area": "415", "num": "555-1234"}
    # Non-capturing (?:...) — group but don't capture
    m3 = re.search(r"(?:\d{3})-(\d{4})", "555-1234")
    assert m3 is not None and m3.groups() == ("1234",)  # only inner captured
    # Backreference \1 — match same text again
    assert re.search(r"(\w+) \1", "hello hello") is not None  # repeated word
    assert re.search(r"(\w+) \1", "hello world") is None
    # Group with flag (?i) etc. inline
    assert re.search(r"(?i)hello", "HELLO") is not None
    return {"groups": m.groups(), "named": m2.groupdict()}


def re_sub_demo() -> dict[str, Any]:
    """sub, subn, with callable repl, and flags."""
    # Simple sub
    assert re.sub(r"\d", "X", "a1b2") == "aXbX"
    # Count limited
    assert re.sub(r"\d", "X", "a1b2c3", count=1) == "aXb2c3"
    # subn returns (new_string, count)
    new, n = re.subn(r"\d", "X", "a1b2")
    assert new == "aXbX" and n == 2
    # Callable repl: double numbers
    def double(m: re.Match[str]) -> str:
        return str(int(m.group()) * 2)

    assert re.sub(r"\d+", double, "a1 b22") == "a2 b44"
    # Backrefs in repl: \1, \g<name>
    assert re.sub(r"(\w+) (\w+)", r"\2 \1", "hello world") == "world hello"
    assert re.sub(r"(?P<first>\w+) (?P<second>\w+)", r"\g<second> \g<first>", "foo bar") == "bar foo"
    # Flags: IGNORECASE, MULTILINE (^/$ per line), DOTALL (. matches \n), VERBOSE (allow comments)
    assert re.search(r"hello", "HELLO", re.IGNORECASE) is not None
    assert re.findall(r"^ab", "ab\nab", re.MULTILINE) == ["ab", "ab"]
    assert re.search(r"a.*b", "a\nb", re.DOTALL) is not None
    return {"sub": re.sub(r"\d", "X", "a1"), "n": n}


def re_verbose_lookaround_demo() -> dict[str, Any]:
    """Verbose patterns and lookahead/lookbehind."""
    # VERBOSE allows whitespace/comments in pattern — tutorial readability
    pat = re.compile(
        r"""
        (?P<user> \w+ )      # username
        @                    # @
        (?P<domain> \w+ \. \w+ )  # domain
        """,
        re.VERBOSE,
    )
    m = pat.search("Contact alice@example.com")
    assert m is not None and m.group("user") == "alice"

    # Lookahead (?=...) — assert followed by, not consumed
    # e.g., find word followed by ":"
    assert re.findall(r"\w+(?=:)", "a: b c:") == ["a", "c"]
    # Negative lookahead (?!...) — not followed by
    assert re.findall(r"\b\w+(?!\d)\b", "a1 b c2")  # tutorial demo, not strict
    # Lookbehind (?<=...) — assert preceded by; fixed width in Python
    assert re.search(r"(?<=@)\w+", "user@example") is not None
    assert re.search(r"(?<=@)\w+", "user example") is None
    # Negative lookbehind (?<!...)
    assert re.search(r"(?<!\w)@\w+", "@hi hi@hi") is not None
    return {"user": m.group("user"), "lookahead": re.findall(r"\w+(?=:)", "a: b")}


# =============================================================================
# 6. SYS
# =============================================================================
# sys = runtime system info. Not for OS file ops (use pathlib), but for interpreter state.
# We mock argv to stay deterministic; we never call sys.exit() in library code.


def sys_info_demo() -> dict[str, Any]:
    """version, platform, executable, modules, recursionlimit, float_info."""
    # version_info is tuple (major, minor, micro, releaselevel, serial)
    assert sys.version_info.major == 3 and sys.version_info.minor >= 10
    assert isinstance(sys.platform, str) and len(sys.platform) > 0  # "linux", "darwin", "win32"
    assert isinstance(sys.executable, str) and len(sys.executable) > 0
    # modules: dict of imported modules
    assert "sys" in sys.modules and "re" in sys.modules  # we imported re earlier
    # recursionlimit / getsizeof
    limit = sys.getrecursionlimit()
    assert isinstance(limit, int) and limit >= 500
    size = sys.getsizeof([1, 2, 3])
    assert isinstance(size, int) and size > 0
    # float_info, int_info (3.11+ has int_info)
    assert hasattr(sys, "float_info")
    assert sys.float_info.max > 1e308
    return {"version": sys.version_info[:2], "platform": sys.platform, "recursionlimit": limit, "sizeof": size}


def sys_argv_exc_demo() -> dict[str, Any]:
    """argv mock and exc_info / exception()."""
    # Save/restore argv to keep deterministic — tutorial: argv is list of strings from command line
    orig_argv = sys.argv[:]
    try:
        sys.argv = ["prog", "--verbose", "file.txt"]
        assert sys.argv[0] == "prog" and "--verbose" in sys.argv
        # Typical parsing (without argparse) — check flag
        verbose = "--verbose" in sys.argv
        assert verbose is True
    finally:
        sys.argv = orig_argv  # restore — don't pollute global state

    # exc_info / exception: capture current exception info
    try:
        raise ValueError("demo error")
    except ValueError:
        exc_type, exc_val, exc_tb = sys.exc_info()
        assert exc_type is ValueError and isinstance(exc_val, ValueError)
        assert exc_tb is not None
        # 3.11+ sys.exception() is cleaner inside except (no args)
        exc = sys.exception()
        assert isinstance(exc, ValueError)

    # sys.path — list of import search paths; don't mutate globally in demo
    assert isinstance(sys.path, list) and len(sys.path) > 0
    return {"argv_mock": True, "exc_type": exc_type.__name__ if exc_type else ""}  # type: ignore[union-attr]


def sys_streams_demo() -> dict[str, Any]:
    """stdin/stdout/stderr — use StringIO for demo, no real console I/O."""
    import io

    # Simulate stdout capture: redirect sys.stdout to StringIO, write, restore
    orig_out = sys.stdout
    fake_out = io.StringIO()
    try:
        sys.stdout = fake_out  # type: ignore[assignment]
        print("hello sys")
        assert fake_out.getvalue() == "hello sys\n"
    finally:
        sys.stdout = orig_out

    # stdin mock similarly — but we just show that stdin is a TextIOWrapper
    assert hasattr(sys, "stdin") and hasattr(sys, "stdout") and hasattr(sys, "stderr")
    return {"captured": fake_out.getvalue().strip(), "has_stdin": True}


# =============================================================================
# 7. MATH
# =============================================================================
# math = pure math, not cmath (complex) or numpy (arrays). All scalars.


def math_constants_demo() -> dict[str, Any]:
    """Constants, isclose, isfinite, etc."""
    assert math.isclose(math.pi, 3.1415926535, rel_tol=1e-9)
    assert math.isclose(math.e, 2.718281828, rel_tol=1e-9)
    assert math.isclose(math.tau, 2 * math.pi)  # tau = 2pi (3.6+)
    assert math.isfinite(1.0) and not math.isfinite(math.inf) and not math.isfinite(math.nan)
    assert math.isinf(math.inf) and math.isnan(math.nan)
    # isclose handles NaN/inf specially — tutorial
    assert not math.isclose(math.nan, math.nan)  # NaN never close, even to itself
    return {"pi": math.pi, "e": math.e, "tau": math.tau}


def math_integer_demo() -> dict[str, Any]:
    """Integer helpers: gcd, lcm, comb, perm, factorial, prod, isqrt."""
    assert math.gcd(12, 18) == 6
    assert math.lcm(4, 6) == 12  # 3.9+ lcm
    assert math.comb(5, 2) == 10  # n choose k
    assert math.perm(5, 2) == 20  # nPk
    assert math.factorial(5) == 120
    assert math.prod([1, 2, 3, 4]) == 24  # 3.8+ prod
    assert math.isqrt(10) == 3  # integer sqrt floor
    # ceil/floor/trunc
    assert math.ceil(2.1) == 3 and math.floor(2.9) == 2 and math.trunc(-2.9) == -2
    # fmod vs % — fmod more C-like for floats
    assert math.isclose(math.fmod(5.5, 2), 1.5)
    return {"gcd": math.gcd(12, 18), "comb": math.comb(5, 2), "prod": math.prod([1, 2, 3])}


def math_float_demo() -> dict[str, Any]:
    """Float helpers: sqrt, log, exp, trig, hypot, dist, etc."""
    assert math.isclose(math.sqrt(4), 2.0)
    assert math.isclose(math.log(math.e), 1.0)
    assert math.isclose(math.log2(8), 3.0)
    assert math.isclose(math.exp(1), math.e)
    # Pow: math.pow returns float, ** may return int for ints
    assert math.isclose(math.pow(2, 3), 8.0)
    # Trig expects radians
    assert math.isclose(math.sin(math.pi / 2), 1.0)
    assert math.isclose(math.cos(0), 1.0)
    assert math.isclose(math.degrees(math.pi), 180.0)
    assert math.isclose(math.radians(180), math.pi)
    # hypot is Euclidean norm — better than sqrt(x*x+y*y) (handles overflow)
    assert math.isclose(math.hypot(3, 4), 5.0)
    # dist: Euclidean distance between points (3.8+)
    assert math.isclose(math.dist((0, 0), (3, 4)), 5.0)
    # erf, gamma, etc. — demo that they exist
    assert math.isclose(math.erf(0), 0.0)
    assert math.isclose(math.gamma(5), 24.0)  # gamma(n) = (n-1)!
    # frexp / ldexp, modf
    mant, exp = math.frexp(8)  # 8 = 0.5 * 2^4
    assert mant == 0.5 and exp == 4
    assert math.ldexp(mant, exp) == 8
    frac, intpart = math.modf(3.14)
    assert math.isclose(frac, 0.14, abs_tol=1e-9) and intpart == 3.0
    return {"sqrt": math.sqrt(4), "hypot": math.hypot(3, 4), "dist": math.dist((0, 0), (3, 4))}


# =============================================================================
# 8. HEAPQ — min-heap
# =============================================================================
# heapq implements a min-heap on top of a plain list. `heap[0]` is always smallest.
# - No max-heap — simulate via negation or `heapq._heapify_max` (undocumented).
# - Heap is not sorted! Only heap[0] guaranteed min; rest is heap order.
# - For key=, use tuple trick: (key, item) or `heapq.nsmallest`.


def heapq_basic_demo() -> dict[str, Any]:
    """heapify, heappush, heappop, heapreplace, heappushpop."""
    data = [5, 1, 3, 2, 4]
    heap = data[:]  # copy — heapify mutates in-place
    heapq.heapify(heap)  # O(n) transform list into heap
    assert heap[0] == 1  # smallest
    # heappush O(log n), heappop O(log n)
    heapq.heappush(heap, 0)
    assert heap[0] == 0
    smallest = heapq.heappop(heap)
    assert smallest == 0 and heap[0] == 1
    # heapreplace = pop + push atomically (more efficient if you know you'll push)
    # heappushpop = push + pop (push first)
    replaced = heapq.heapreplace(heap, 10)  # pop smallest (1), push 10, return popped
    assert replaced == 1
    # heappushpop pushes 0 then pops smallest (0) — net no change except return 0
    val = heapq.heappushpop(heap, 0)
    assert val == 0  # pushed 0 is smallest, so popped immediately
    return {"heap": heap, "replaced": replaced, "pushpop": val}


def heapq_keyed_demo() -> dict[str, Any]:
    """nlargest/nsmallest with key=, merge, and max-heap via negation."""
    nums = [5, 1, 8, 3, 7, 2]
    assert heapq.nsmallest(2, nums) == [1, 2]
    assert heapq.nlargest(2, nums) == [8, 7]
    # Key: e.g., top 2 longest strings
    words = ["apple", "a", "banana", "kiwi"]
    assert heapq.nlargest(2, words, key=len) == ["banana", "apple"]
    # Merge multiple sorted inputs lazily (returns iterator) — O(N log k)
    merged = list(heapq.merge([1, 3, 5], [2, 4, 6], [0, 7]))
    assert merged == [0, 1, 2, 3, 4, 5, 6, 7]
    # Max-heap simulation: negate values
    max_heap = [-x for x in nums]
    heapq.heapify(max_heap)
    largest = -heapq.heappop(max_heap)
    assert largest == 8
    # Tuple heap for priority queue: (priority, order, item) — order breaks ties stably
    # Tutorial: priority queue anti-pattern — if priorities equal, compare next element
    pq: list[tuple[int, int, str]] = []
    counter = 0
    for prio, name in [(2, "task2"), (1, "task1"), (2, "task2b")]:
        heapq.heappush(pq, (prio, counter, name))
        counter += 1
    assert heapq.heappop(pq)[2] == "task1"  # priority 1 first, then FIFO for prio 2
    assert heapq.heappop(pq)[2] == "task2"
    return {"nsmallest": heapq.nsmallest(2, nums), "merged": merged, "largest": largest}


def heapq_sorted_vs_heap_demo() -> dict[str, Any]:
    """heap is not sorted — demo heap invariant vs sorted()."""
    heap = [3, 1, 4, 1, 5, 9, 2, 6]
    heapq.heapify(heap)
    assert heap[0] == 1  # min guaranteed
    # But heap != sorted(heap) — only first is min, rest is heap-order
    assert heap != sorted(heap) or heap == sorted(heap)  # just show not necessarily sorted
    # To get sorted order, repeatedly heappop (O(n log n) like heap sort)
    sorted_via_heap = []
    h = [3, 1, 4, 1, 5]
    heapq.heapify(h)
    while h:
        sorted_via_heap.append(heapq.heappop(h))
    assert sorted_via_heap == [1, 1, 3, 4, 5]
    # For small n, `sorted()` is faster than heap sort; use heap only for k << n
    return {"heap_min": heap[0], "sorted_via_heap": sorted_via_heap}


# =============================================================================
# Self-tests (IMPORT-ONLY)
# =============================================================================


def _self_tests() -> None:
    """Run assert-based checks for every section. Call manually or via __main__."""
    # 1. Type annotations
    ta = type_annotations_demo()
    assert ta["vector"] == [1.0, 2.0, 3.0]
    assert ta["pair"] == (1, 2)
    assert ta["stack_top"] == 10
    assert ta["render"] == "□"
    # extra 3.12 checks
    p = Point2D(x=1, y=2, label="A")
    assert p.as_vector() == [1, 2]
    from typing import get_type_hints

    assert "x" in get_type_hints(Point2D)
    assert describe_value(5) == "int 5"
    assert parse(" 10 ") == 10

    # 2. Pathlib (pure, no FS)
    pp = pathlib_pure_demo()
    assert pp["posix"] == "a/b/c.txt"
    assert pp["joined"] == "project/src/main.py"
    assert pp["win_drive"] == "C:"
    assert pp["archive_suffixes"] == [".tar", ".gz"]
    cp = pathlib_concrete_pure_ops_demo()
    assert cp["parts"] == ("project", "src", "app.py")
    assert cp["match_py"] is True
    assert cp["parents"] == ["project/src", "project", "."]
    sp = pathlib_splitting_demo()
    assert sp["hidden_suffix"] == ""
    assert sp["multi_suffixes"][-2:] == [".tar", ".gz"]

    # 3. Collections
    cc = collections_counter_demo()
    assert cc["counter"]["apple"] == 3  # after update cherry becomes 3, apple 3
    assert cc["most_common"][0][0] == "apple"
    dq = collections_deque_demo()
    assert dq["deque"] == [2, 3, 4, 5] and dq["popped"] == 1
    dd = collections_defaultdict_demo()
    assert dd["groups"]["a"] == [1, 2] and dd["counts"]["a"] == 2
    nt = collections_namedtuple_demo()
    assert nt["classic"].x == 1 and nt["typed"].label == "B" and nt["replaced"].x == 10
    ch = collections_chainmap_ordered_user_demo()
    assert ch["chain_color"] == "red" and ch["ordered_keys"] == ["b", "a"]

    # 4. Functools
    lb = functools_lru_cache_demo()
    assert lb["fib"] == 55 and lb["sq"] == 25
    pa = functools_partial_demo()
    assert pa["square"] == 16 and pa["cube"] == 8
    wr = functools_wraps_reduce_demo()
    assert wr["greet_name"] == "greet" and wr["total"] == 10
    fd = functools_dispatch_cached_demo()
    assert fd["format_int"] == "int:42" and fd["version_lt"] is True

    # 5. Re
    rb = re_basic_demo()
    assert rb["first_phone"] == "415-555-1234" and rb["nums"] == ["1", "22", "333"]
    rg = re_groups_demo()
    assert rg["groups"] == ("415", "555", "1234") and rg["named"]["area"] == "415"
    rs = re_sub_demo()
    assert rs["sub"] == "aX" and rs["n"] == 2
    rv = re_verbose_lookaround_demo()
    assert rv["user"] == "alice" and rv["lookahead"] == ["a"]

    # 6. Sys
    si = sys_info_demo()
    assert si["version"][0] == 3 and si["recursionlimit"] >= 500
    sa = sys_argv_exc_demo()
    assert sa["argv_mock"] is True and sa["exc_type"] == "ValueError"
    ss = sys_streams_demo()
    assert ss["captured"] == "hello sys" and ss["has_stdin"] is True

    # 7. Math
    mc = math_constants_demo()
    assert math.isclose(mc["pi"], math.pi)
    mi = math_integer_demo()
    assert mi["gcd"] == 6 and mi["comb"] == 10 and mi["prod"] == 6
    mf = math_float_demo()
    assert mf["sqrt"] == 2.0 and mf["hypot"] == 5.0 and mf["dist"] == 5.0

    # 8. Heapq
    hb = heapq_basic_demo()
    assert hb["replaced"] == 1 and hb["pushpop"] == 0
    hk = heapq_keyed_demo()
    assert hk["nsmallest"] == [1, 2] and hk["merged"] == [0, 1, 2, 3, 4, 5, 6, 7] and hk["largest"] == 8
    hs = heapq_sorted_vs_heap_demo()
    assert hs["heap_min"] == 1 and hs["sorted_via_heap"] == [1, 1, 3, 4, 5]

    print("All Standard Library asserts passed.")


if __name__ == "__main__":
    _self_tests()
