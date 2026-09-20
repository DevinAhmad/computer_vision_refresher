"""
Tier 3 — Python Depth
======================

IMPORT-ONLY module.  No code runs on import; run tests manually:

    import tier3_python_depth
    tier3_python_depth._self_tests()        # sync checks
    # async checks:
    import asyncio
    asyncio.run(tier3_python_depth._async_self_tests())

    # or
    python tier3_python_depth.py            # runs both sync + async

Topics covered (numbered as per prompt 21-28)
----------------------------------------------
21. Iterators / Generators
22. Decorators
23. OOP (classes, inheritance, dunder methods, dataclasses, properties)
24. Exceptions (custom, chaining, context managers)
25. Memory / Reference Behavior (mutable defaults, shallow vs deep copy, is vs ==)
26. Type Hints (Optional, Union, Callable, Generic, Protocol)
27. asyncio (async/await, gather, create_task, cancellation)
28. Concurrency (threading, multiprocessing, concurrent.futures, GIL notes)

Every section contains verbose inline comments explaining *why* a construct
behaves the way it does — suitable for interview follow-up questions
(“what happens if you forget functools.wraps?” etc.).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import functools
import multiprocessing
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, Protocol, TypeVar, Union

# =============================================================================
# 21. ITERATORS / GENERATORS
# =============================================================================
# Iterator protocol: object with __iter__() -> iterator and __next__() -> value
# or raises StopIteration. Generator functions use `yield` and automatically
# implement the iterator protocol. Generators are lazy — they produce values
# on demand and keep local state between yields.


class Countdown:
    """
    Manual iterator: counts down from n to 1.

    Demonstrates the full iterator protocol without `yield`.
    - __iter__ returns self (iterator must be iterable)
    - __next__ returns next value or raises StopIteration
    - __iter__ returning self means single-pass; for multi-pass return a new iterator
    """

    def __init__(self, n: int):
        self.n = n
        self.current = n

    def __iter__(self) -> Iterator[int]:
        # Reset on new iteration so `for x in Countdown(3)` works repeatedly
        # only if we return a fresh iterator. Here we reset self for simplicity.
        # Production code would often return CountdownIterator(self.n) instead.
        self.current = self.n
        return self

    def __next__(self) -> int:
        if self.current <= 0:
            raise StopIteration  # signals exhaustion to `for` loop
        val = self.current
        self.current -= 1
        return val


def fib_generator(n: int) -> Generator[int, None, None]:
    """
    Generator function: yields first n Fibonacci numbers lazily.

    Each `yield` suspends the function, saving locals (a, b, count).
    Next call resumes right after `yield`. Memory O(1) vs O(n) for list version.
    Time O(n) to generate all, but values are produced one-by-one.
    """
    a, b = 0, 1
    for _ in range(n):
        yield a  # suspend here; caller receives `a`
        a, b = b, a + b  # update after resume


def generator_expression_demo(nums: list[int]) -> int:
    """
    Generator expression vs list comprehension.

    (x*x for x in nums) — lazy, O(1) memory, computes on iteration.
    [x*x for x in nums] — eager, O(n) memory, computes immediately.
    Use generator expressions when you only need to iterate once.
    """
    # Generator expression — not yet computed! Wrapped in sum() which iterates it.
    gen = (x * x for x in nums if x % 2 == 0)
    # We can check it's a generator: hasattr(gen, '__next__')
    return sum(gen)  # consumes lazily


def yield_from_demo() -> Generator[int, None, None]:
    """
    `yield from` delegates to a sub-generator.

    Equivalent to `for val in fib_generator(5): yield val` but more efficient
    and preserves .send()/.throw() delegation. Useful for composing generators.
    """

    def inner() -> Generator[int, None, None]:
        yield from fib_generator(5)  # delegate to fib

    yield from inner()  # double delegation for demo
    # After inner exhausted, continue
    yield 99


def infinite_counter(start: int = 0) -> Generator[int, None, None]:
    """
    Infinite generator — must be consumed with care (e.g., islice or break).

    Demonstrates that generators can represent infinite sequences without
    infinite memory. Caller decides when to stop.
    """
    n = start
    while True:
        yield n
        n += 1


# =============================================================================
# 22. DECORATORS
# =============================================================================
# Decorators are higher-order functions that wrap another function/class to
# add behavior without modifying its source. Key pitfalls: forgetting
# @functools.wraps loses __name__/__doc__, stacking order matters.


def simple_logger(func: Callable) -> Callable:
    """
    Simple decorator that logs calls. Shows the basic closure pattern:

        @simple_logger
        def foo(): ...

    is syntax sugar for:  foo = simple_logger(foo)

    Without @functools.wraps, foo.__name__ would become 'wrapper' — breaks
    introspection, debuggers, and stacked decorators.
    """

    @functools.wraps(func)  # copy __name__, __doc__, __wrapped__, etc.
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Verbose: log before and after — useful for tracing
        # print(f"[LOG] calling {func.__name__} args={args} kwargs={kwargs}")
        result = func(*args, **kwargs)
        # print(f"[LOG] {func.__name__} returned {result!r}")
        return result

    return wrapper


def retry(max_attempts: int = 3, delay: float = 0.0) -> Callable:
    """
    Parameterized decorator (decorator factory) — takes args, returns decorator.

    Usage: @retry(max_attempts=5) — needs extra nesting: factory -> decorator -> wrapper.
    Retries function if it raises an exception, up to max_attempts times.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 — demo retry on any exception
                    last_exc = exc
                    if attempt < max_attempts and delay:
                        time.sleep(delay)
                    # Verbose: we swallow last_exc until final failure
            # All attempts failed — re-raise last exception preserving traceback
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


def count_calls(func: Callable) -> Callable:
    """
    Decorator that counts how many times the function was called.

    Demonstrates attaching state to wrapper via attribute. Alternative is to
    use a class-based decorator.
    """
    # Initialize counter on wrapper after definition; use closure to hold it
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wrapper.calls += 1  # type: ignore[attr-defined]
        return func(*args, **kwargs)

    wrapper.calls = 0  # type: ignore[attr-defined]
    return wrapper


class ClassDecorator:
    """
    Class-based decorator — useful when you need richer state or to decorate classes.

    __call__ makes instances callable, so ClassDecorator can be used as @ClassDecorator.
    Shows that decorators don't have to be functions.
    """

    def __init__(self, func: Callable):
        functools.update_wrapper(self, func)  # mimic wraps for class
        self.func = func
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.func(*args, **kwargs)


def cache_memoize(func: Callable) -> Callable:
    """
    Simple memoization decorator (like functools.lru_cache but minimal).

    Caches results keyed by args. Only works for hashable args.
    Shows manual cache dict vs using functools.lru_cache.
    """
    cache: dict[tuple[Any, ...], Any] = {}

    @functools.wraps(func)
    def wrapper(*args: Any) -> Any:
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    wrapper.cache = cache  # type: ignore[attr-defined]  — expose for introspection/tests
    return wrapper


# =============================================================================
# 23. OOP — Classes, Inheritance, Dunders, Dataclasses, Properties
# =============================================================================


class Animal(ABC):
    """
    Abstract base class — cannot be instantiated directly.

    @abstractmethod enforces that subclasses implement `speak`.
    Shows inheritance, polymorphism, and ABC enforcement.
    """

    def __init__(self, name: str):
        self.name = name  # instance attribute

    @abstractmethod
    def speak(self) -> str:
        """Subclasses must implement."""

    def describe(self) -> str:
        """Concrete method — inherited as-is."""
        return f"{self.name} is an animal"


class Dog(Animal):
    """
    Concrete subclass of Animal. Demonstrates inheritance, super(), and
    overriding + dunder methods (__str__, __repr__, __eq__, __len__ analog).
    """

    species = "Canis familiaris"  # class attribute — shared across instances

    def __init__(self, name: str, breed: str):
        super().__init__(name)  # call parent __init__ to set self.name
        self.breed = breed

    def speak(self) -> str:
        return f"{self.name} says Woof!"

    def __str__(self) -> str:
        # Human-readable, for print()/str()
        return f"Dog(name={self.name}, breed={self.breed})"

    def __repr__(self) -> str:
        # Unambiguous, ideally eval-able
        return f"Dog(name={self.name!r}, breed={self.breed!r})"

    def __eq__(self, other: object) -> bool:
        # Equality based on name+breed — not identity (is)
        if not isinstance(other, Dog):
            return NotImplemented
        return self.name == other.name and self.breed == other.breed


@dataclass
class Point:
    """
    Dataclass — auto-generates __init__, __repr__, __eq__, etc.

    Shows: default values, field(default_factory) for mutable defaults,
    ordering, frozen (immutability), and custom methods.
    """

    x: float
    y: float
    label: str = ""  # default value — safe because str is immutable
    # For mutable defaults, use field(default_factory=list) — NOT tags: list = []
    tags: list[str] = field(default_factory=list)

    def distance_from_origin(self) -> float:
        """Instance method using self."""
        return (self.x**2 + self.y**2) ** 0.5

    def __add__(self, other: Point) -> Point:
        """Dunder for `p1 + p2` — returns new Point."""
        if not isinstance(other, Point):
            return NotImplemented
        return Point(self.x + other.x, self.y + other.y)


@dataclass(frozen=True, order=True)
class FrozenPoint:
    """
    Frozen (immutable) + ordered dataclass.

    Frozen: instances hashable, can be used in sets/dicts. Attempts to mutate
    raise FrozenInstanceError. Order: auto-generates __lt__ etc. based on fields.
    """

    x: float
    y: float


class Temperature:
    """
    Property demo: Celsius <-> Fahrenheit conversion with validation.

    @property makes method act like attribute access.
    @<name>.setter defines assignment behavior with validation.
    """

    def __init__(self, celsius: float = 0.0):
        self._celsius = celsius  # “private” by convention (single underscore)

    @property
    def celsius(self) -> float:
        """Getter — accessed as `t.celsius` without parentheses."""
        return self._celsius

    @celsius.setter
    def celsius(self, value: float) -> None:
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        """Computed property — no setter means read-only."""
        return self._celsius * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        # Delegate to celsius setter for validation
        self.celsius = (value - 32) * 5 / 9


class CounterWithClassMethods:
    """
    Demonstrates @classmethod vs @staticmethod vs instance method.

    - Instance method: needs self, operates on instance.
    - @classmethod: receives cls, operates on class (factory pattern).
    - @staticmethod: no self/cls, plain function in class namespace.
    """

    count = 0  # class variable

    def __init__(self, value: int):
        self.value = value
        CounterWithClassMethods.count += 1

    @classmethod
    def from_string(cls, s: str) -> CounterWithClassMethods:
        """Alternative constructor — receives class, not instance."""
        return cls(int(s))

    @staticmethod
    def is_valid_value(v: int) -> bool:
        """Utility — doesn't need self or cls."""
        return isinstance(v, int) and v >= 0

    @classmethod
    def get_count(cls) -> int:
        return cls.count


# =============================================================================
# 24. EXCEPTIONS
# =============================================================================
# Exceptions are objects; `raise` creates them, `try/except` handles them.
# Best practices: catch specific exceptions, use `raise ... from` for chaining,
# always clean up with `finally` or context managers.


class AppError(Exception):
    """Base application error — inherits from Exception (not BaseException)."""

    pass


class ValidationError(AppError):
    """Specific error with extra context — shows custom exception with data."""

    def __init__(self, message: str, field: str = ""):
        super().__init__(message)
        self.field = field  # extra attribute for callers to inspect


def parse_positive_int(s: str) -> int:
    """
    Parse string to positive int — demonstrates try/except/else/finally + raise from.

    - try: attempt risky operation
    - except ValueError: handle specific error, chain with `from`
    - else: runs only if NO exception (success path)
    - finally: always runs (cleanup) — even if return/raised inside try
    """
    try:
        value = int(s)  # may raise ValueError
    except ValueError as exc:
        # Chain: preserve original traceback as __cause__
        raise ValidationError(f"Not an integer: {s!r}", field="value") from exc
    else:
        # Only runs if int() succeeded
        if value <= 0:
            raise ValidationError(f"Must be positive, got {value}", field="value")
        return value
    finally:
        # Always runs — good for cleanup; here just illustrative
        pass  # e.g., close file, release lock


def safe_divide(a: float, b: float) -> Optional[float]:
    """
    Demonstrate catching specific exception and returning fallback.

    Catches ZeroDivisionError only — not broad `except Exception` which hides bugs.
    """
    try:
        return a / b
    except ZeroDivisionError:
        return None  # caller must handle None


@contextmanager
def managed_resource(name: str) -> Generator[str, None, None]:
    """
    Context manager via @contextmanager — ensures cleanup even on exception.

    `with managed_resource("db") as res:` will always execute `finally` block.
    Equivalent to writing a class with __enter__/__exit__.
    """
    # Setup
    resource = f"resource:{name}"
    # print(f"Acquiring {resource}")
    try:
        yield resource  # body of `with` executes here
    finally:
        # Cleanup — runs even if body raised exception
        # print(f"Releasing {resource}")
        pass


class FileLikeContext:
    """
    Manual context manager class — shows __enter__/__exit__ protocol.

    __exit__ receives exception info (type, value, traceback) or (None,None,None)
    if no exception. Returning True suppresses the exception.
    """

    def __init__(self, name: str):
        self.name = name
        self.entered = False
        self.exited = False

    def __enter__(self) -> FileLikeContext:
        self.entered = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.exited = True
        # Return False (or None) to propagate exception; True to suppress
        return False


# =============================================================================
# 25. MEMORY / REFERENCE BEHAVIOR
# =============================================================================
# Python variables are references (names pointing to objects). Assignment
# never copies; `is` checks identity (same object), `==` checks equality
# (value). Mutable objects (list, dict, set) shared references cause aliasing bugs.


def mutable_alias_demo() -> dict[str, Any]:
    """
    Show aliasing vs copying.

    `b = a` does NOT copy — both names point to same list object.
    Mutating via one name is visible through the other.
    """
    a = [1, 2, 3]
    b = a  # alias — same object (id(a) == id(b))
    c = a[:]  # shallow copy via slice — new object, same elements
    d = copy.copy(a)  # explicit shallow copy
    b.append(4)  # mutates the shared object
    # Now a == [1,2,3,4] as well! But c and d remain [1,2,3]
    return {"a": a, "b": b, "c": c, "d": d, "a_is_b": a is b, "a_is_c": a is c}


def shallow_vs_deep_copy_demo() -> dict[str, Any]:
    """
    Shallow copy copies outer list but inner objects still shared.
    Deep copy recursively copies all nested objects.

    Crucial for nested mutables: shallow copy of [[1,2],[3,4]] shares inner lists.
    """
    original = [[1, 2], [3, 4]]
    shallow = copy.copy(original)  # or original[:]
    deep = copy.deepcopy(original)

    # Mutate inner list via original
    original[0].append(99)
    # shallow[0] sees the mutation (shared inner list), deep[0] does not
    original.append([5, 6])  # mutate outer — shallow/deep unaffected

    return {
        "original": original,  # [[1,2,99],[3,4],[5,6]]
        "shallow": shallow,  # [[1,2,99],[3,4]]  — inner mutated!
        "deep": deep,  # [[1,2],[3,4]]       — isolated
        "shallow_inner_is_original": shallow[0] is original[0],
        "deep_inner_is_original": deep[0] is original[0],
    }


def mutable_default_trap_demo() -> Callable[[Any], list[Any]]:
    """
    Return a pair of functions showing the mutable default argument trap and fix.

    WRONG: def append(item, lst=[]):  # one list shared across ALL calls
    RIGHT: def append(item, lst=None): if lst is None: lst = []
    """

    # WRONG version — for demonstration (we isolate it so it doesn't pollute tests)
    def wrong_append(item: Any, lst: list[Any] = []) -> list[Any]:  # noqa: B006  # intentional trap demo
        lst.append(item)
        return lst

    # RIGHT version
    def right_append(item: Any, lst: Optional[list[Any]] = None) -> list[Any]:
        if lst is None:
            lst = []
        lst.append(item)
        return lst

    # Attach both for testing; caller can inspect
    right_append.wrong = wrong_append  # type: ignore[attr-defined]
    return right_append  # type: ignore[return-value]


def is_vs_equality_demo() -> dict[str, Any]:
    """
    `is` checks identity (same object in memory); `==` checks equality (value).

    CPython interns small ints (-5..256) and string literals, so `is` may
    appear to work for small numbers — but don't rely on it! Always use `==`
    for value comparison. Use `is` only for None/singleton checks (`x is None`).
    """
    a = [1, 2, 3]
    b = [1, 2, 3]
    c = a
    # a == b True (same values), a is b False (different objects), a is c True
    # Demonstrate interning pitfall:
    x = 1000
    y = 1000  # may be same or different object depending on context — don't rely on `is`
    return {
        "a_eq_b": a == b,
        "a_is_b": a is b,
        "a_is_c": a is c,
        "x_eq_y": x == y,
        # x is y is unreliable — we just show eq is the correct check
        "none_check": (None is None),  # True — singleton
    }


def reference_counting_demo() -> int:
    """
    Show that multiple references keep an object alive.

    In CPython, refcount determines when memory is freed (plus GC for cycles).
    This demo just shows id() stays same when aliasing.
    """
    obj = {"key": "value"}
    ref1 = obj
    ref2 = obj
    # All three names point to same dict object
    assert ref1 is obj and ref2 is obj
    assert id(obj) == id(ref1) == id(ref2)
    return id(obj)


# =============================================================================
# 26. TYPE HINTS
# =============================================================================
# Type hints are annotations for static checkers (mypy, pyright). They don't
# affect runtime (except via typing.get_type_hints). Use `from __future__ import
# annotations` to make all annotations strings (lazy evaluation, forward refs).


T = TypeVar("T")  # generic type variable
Number = Union[int, float]  # Union — value can be int OR float


def typed_greet(name: str, age: int) -> str:
    """Simple typed function: inputs and return type annotated."""
    return f"{name} is {age} years old"


def optional_demo(value: Optional[str] = None) -> str:
    """
    Optional[X] is shorthand for Union[X, None] — value may be None.

    Always handle the None case to satisfy type checkers.
    """
    if value is None:
        return "No value"
    return value.upper()  # type checker knows value is str here (narrowing)


def callable_demo(func: Callable[[int, int], int], a: int, b: int) -> int:
    """
    Callable[[ArgTypes], ReturnType] — describes a function parameter.

    Callable[[int, int], int] means: function taking two ints, returning int.
    """
    return func(a, b)


def generic_first(items: list[T]) -> Optional[T]:
    """Generic function — works for any type T, preserves type info for checker."""
    return items[0] if items else None


class Stack(Generic[T]):
    """
    Generic class — Stack[int] vs Stack[str] distinguished by type checker.

    Runtime behavior identical; typing just helps catch bugs early.
    """

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> Optional[T]:
        return self._items[-1] if self._items else None


class Drawable(Protocol):
    """
    Protocol — structural subtyping (duck typing for type checkers).

    Any object with a `draw() -> str` method satisfies Drawable, without
    explicit inheritance. PEP 544.
    """

    def draw(self) -> str: ...


def render(obj: Drawable) -> str:
    """Accepts any Drawable — no need to inherit from Drawable base class."""
    return obj.draw()


def union_vs_pipe_demo(value: int | str) -> str:  # PEP 604 modern union syntax (Python 3.10+)
    """
    Modern union syntax: `int | str` instead of `Union[int, str]`.

    Equivalent for type checkers. This function handles either type.
    """
    if isinstance(value, int):
        return f"int:{value}"
    return f"str:{value}"


# =============================================================================
# 27. ASYNCIO
# =============================================================================
# asyncio uses cooperative multitasking: `async def` defines a coroutine,
# `await` yields control to the event loop until the awaited thing completes.
# Unlike threads, only one coroutine runs at a time — no race conditions on
# shared data unless you explicitly await.


async def async_fetch(name: str, delay: float = 0.01, fail: bool = False) -> str:
    """
    Simulated async I/O (e.g., HTTP fetch). Uses asyncio.sleep to yield control.

    `async def` returns a coroutine object when called — must be awaited or
    scheduled as a task. `await` pauses this coroutine, letting others run.
    """
    await asyncio.sleep(delay)  # non-blocking sleep — event loop runs other tasks
    if fail:
        raise AppError(f"Fetch failed for {name}")
    return f"data:{name}"


async def async_gather_demo() -> list[str]:
    """
    Run multiple coroutines concurrently via asyncio.gather.

    `gather` schedules all coroutines concurrently and waits for all.
    Time ~ max(delays) not sum(delays) — true concurrency for I/O-bound work.
    If any raises, gather propagates the exception (unless return_exceptions=True).
    """
    # Schedule 3 fetches concurrently — total time ~ max(0.02, 0.01, 0.015) = 0.02s
    results = await asyncio.gather(
        async_fetch("a", 0.02),
        async_fetch("b", 0.01),
        async_fetch("c", 0.015),
    )
    return results  # ["data:a", "data:b", "data:c"] — order matches gather args


async def async_create_task_demo() -> list[str]:
    """
    Explicit tasks via asyncio.create_task — more control than gather.

    create_task schedules immediately; you can cancel, check done(), etc.
    Gathering tasks after creation achieves same concurrency as gather.
    """
    task1 = asyncio.create_task(async_fetch("x", 0.02))
    task2 = asyncio.create_task(async_fetch("y", 0.01))
    # Both already running concurrently at this point
    # Wait for both — order of await determines result order here
    r1 = await task1
    r2 = await task2
    return [r1, r2]


async def async_timeout_demo() -> str:
    """
    Timeout pattern via asyncio.wait_for.

    wait_for cancels the coroutine if it doesn't complete in time, raising
    TimeoutError (which is asyncio.TimeoutError, subclass of Exception).
    """
    try:
        # This will timeout if fetch takes longer than 0.005s
        return await asyncio.wait_for(async_fetch("slow", 0.05), timeout=0.005)
    except asyncio.TimeoutError:
        return "timed out"


async def async_context_manager_demo() -> str:
    """
    Async context manager (`async with`) — for resources needing async setup/teardown.

    Example: async database connections, aiohttp sessions.
    """

    class AsyncResource:
        async def __aenter__(self) -> str:
            await asyncio.sleep(0.001)  # async setup
            return "resource"

        async def __aexit__(self, *args: Any) -> None:
            await asyncio.sleep(0.001)  # async teardown

    async with AsyncResource() as res:
        await asyncio.sleep(0.001)
        return res


async def async_generator_demo() -> list[int]:
    """
    Async generator — `async def` + `yield`. Consumed via `async for`.

    Useful for streaming async data (e.g., paginated API results).
    """

    async def async_gen(n: int) -> Any:  # AsyncGenerator[int, None]
        for i in range(n):
            await asyncio.sleep(0.001)  # simulate async work per item
            yield i

    result: list[int] = []
    async for val in async_gen(5):  # async for — awaits each yield
        result.append(val)
    return result


# =============================================================================
# 28. CONCURRENCY — threading, multiprocessing, concurrent.futures, GIL
# =============================================================================
# Key distinctions:
# - threading: multiple threads in one process, share memory, GIL limits CPU-bound
#              parallelism (only one thread executes Python bytecode at a time).
#              Great for I/O-bound tasks (network, file) where threads wait.
# - multiprocessing: separate processes, no GIL limit, true parallelism for CPU-bound,
#                    but heavier (pickle overhead, separate memory).
# - concurrent.futures: high-level pool executors for both.
# - GIL (Global Interpreter Lock): CPython lock that protects refcount/GC.
#   Explains why threads don't speed up CPU-bound Python loops — use processes or
#   release GIL via C extensions (numpy) or use free-threaded Python 3.13+.


def threading_demo() -> int:
    """
    Threading with shared counter protected by Lock.

    Without Lock, `counter += 1` is not atomic (read-modify-write) and races
    cause lost increments. Lock ensures mutual exclusion.
    Time is I/O-like (sleep), so threads help despite GIL.
    """
    counter = 0
    lock = threading.Lock()  # ensures only one thread enters critical section

    def worker(n: int) -> None:
        nonlocal counter
        for _ in range(n):
            # Simulate some work
            time.sleep(0.0001)  # I/O-like wait — GIL released
            with lock:  # acquire, increment, release atomically
                counter += 1

    threads = [threading.Thread(target=worker, args=(100,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()  # wait for all to finish
    return counter  # should be 400 if Lock worked


def thread_pool_demo(nums: list[int]) -> list[int]:
    """
    ThreadPoolExecutor — high-level thread pool for I/O-bound tasks.

    Submits tasks, collects via futures. `map` preserves order.
    Ideal when you have many I/O waits (HTTP, file reads).
    """
    def square(n: int) -> int:
        time.sleep(0.001)  # simulate I/O wait
        return n * n

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # executor.map blocks until all done, preserves input order
        results = list(executor.map(square, nums))
    return results


def process_pool_demo(nums: list[int]) -> list[int]:
    """
    ProcessPoolExecutor — true parallelism for CPU-bound work.

    Each task runs in a separate process (pickle args/results). No GIL limit,
    but overhead higher — only worthwhile if work per task is substantial.
    Note: must be guarded by `if __name__ == "__main__"` when run as script
    on spawn-based OS (Windows/macOS). Inside module import it's fine if caller
    guards. Here we keep tasks simple and small for demo.
    """
    # Module-level function needed — lambdas/local funcs can't be pickled
    # So we call the top-level helper `_cpu_square`
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_cpu_square, nums))
    return results


def _cpu_square(n: int) -> int:
    """Top-level helper for process pool (must be picklable — no closures)."""
    # CPU-bound work (no sleep — pure computation)
    total = 0
    for i in range(10000):
        total += (n * i) % 7  # dummy CPU work
    return n * n  # return meaningful result


def queue_thread_safe_demo() -> list[int]:
    """
    Queue for thread-safe communication between threads.

    queue.Queue is thread-safe (internal locks). Use it to pass data between
    producer/consumer threads without manual locking. Different from deque
    which is NOT thread-safe for all ops (though append/popleft are atomic
    in CPython due to GIL, Queue is the correct abstraction).
    """
    import queue

    q: queue.Queue[int] = queue.Queue()
    results: list[int] = []

    def producer() -> None:
        for i in range(5):
            q.put(i)  # thread-safe enqueue

    def consumer() -> None:
        while True:
            try:
                val = q.get(timeout=0.5)  # blocks until item available
                results.append(val * 10)
                q.task_done()
                if val == 4:
                    break
            except queue.Empty:
                break

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return sorted(results)  # sorted for deterministic assert


def gil_demo_explanation() -> dict[str, str]:
    """
    Return textual explanation of GIL for interview answers — not executable.

    Interviewers often ask: “Why doesn’t threading speed up CPU-bound Python?”
    Answer: GIL. This function documents the talking points.
    """
    return {
        "GIL": "Global Interpreter Lock — mutex that allows only one thread to execute Python bytecode at a time in CPython.",
        "CPU-bound threads": "No speedup (may even be slower due to context switching). Use multiprocessing or C extensions that release GIL (numpy, etc.).",
        "IO-bound threads": "Great speedup — threads release GIL while waiting for I/O (network, disk, sleep), so others can run.",
        "multiprocessing": "Separate processes, separate GILs, true parallelism. Overhead: pickling, IPC, higher memory.",
        "alternatives": "asyncio for I/O concurrency (single thread, cooperative), concurrent.futures for pool abstraction, free-threaded Python 3.13+ (experimental no-GIL build).",
        "correctness": "Even with GIL, still need Locks for correctness — GIL doesn't make `counter += 1` atomic (bytecode interleaving).",
    }


# =============================================================================
# Self-tests (IMPORT-ONLY)
# =============================================================================


def _self_tests() -> None:
    """Sync assert-based checks. For async tests see _async_self_tests."""
    # 21. Iterators/generators
    assert list(Countdown(3)) == [3, 2, 1]
    assert list(Countdown(0)) == []
    assert list(fib_generator(5)) == [0, 1, 1, 2, 3]
    assert generator_expression_demo([1, 2, 3, 4]) == (4 + 16)  # 2^2 + 4^2
    assert list(yield_from_demo()) == [0, 1, 1, 2, 3, 99]
    # infinite_counter — take first 3
    import itertools

    assert list(itertools.islice(infinite_counter(10), 3)) == [10, 11, 12]

    # 22. Decorators
    @simple_logger
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert add(2, 3) == 5
    assert add.__name__ == "add"  # wraps preserved name

    attempts = {"count": 0}

    @retry(max_attempts=3)
    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("fail")
        return "ok"

    assert flaky() == "ok" and attempts["count"] == 3

    @count_calls
    def foo() -> None:
        pass

    foo()
    foo()
    assert foo.calls == 2  # type: ignore[attr-defined]

    @ClassDecorator
    def bar(x: int) -> int:
        return x * 2

    assert bar(3) == 6 and bar.calls == 1  # type: ignore[attr-defined]

    @cache_memoize
    def fib(n: int) -> int:
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)  # type: ignore[operator]

    assert fib(10) == 55
    assert (10,) in fib.cache  # type: ignore[attr-defined]  # key is args tuple (10,)

    # 23. OOP
    # Abstract class cannot be instantiated
    try:
        Animal("generic")  # type: ignore[abstract]
        assert False, "should have raised TypeError"
    except TypeError:
        pass
    dog = Dog("Rex", "Labrador")
    assert dog.speak() == "Rex says Woof!"
    assert str(dog) == "Dog(name=Rex, breed=Labrador)"
    assert dog == Dog("Rex", "Labrador")
    assert dog != Dog("Rex", "Poodle")
    # Dataclass
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    p3 = p1 + p2
    assert p3 == Point(4, 6)
    assert FrozenPoint(1, 2) == FrozenPoint(1, 2)
    # Frozen immutability
    fp = FrozenPoint(1, 2)
    try:
        fp.x = 99  # type: ignore[misc]
        assert False, "should have raised FrozenInstanceError"
    except Exception:  # dataclasses.FrozenInstanceError
        pass
    # Property
    t = Temperature(25)
    assert t.celsius == 25 and round(t.fahrenheit, 1) == 77.0
    t.fahrenheit = 32
    assert t.celsius == 0
    try:
        t.celsius = -300
        assert False
    except ValueError:
        pass
    # Classmethod/staticmethod
    c1 = CounterWithClassMethods(5)
    c2 = CounterWithClassMethods.from_string("10")
    assert c2.value == 10 and CounterWithClassMethods.is_valid_value(5) is True

    # 24. Exceptions
    assert parse_positive_int("42") == 42
    try:
        parse_positive_int("abc")
        assert False
    except ValidationError as e:
        assert e.field == "value" and e.__cause__ is not None
    try:
        parse_positive_int("-5")
        assert False
    except ValidationError:
        pass
    assert safe_divide(10, 2) == 5
    assert safe_divide(10, 0) is None
    # Context managers
    with managed_resource("test") as res:
        assert res == "resource:test"
    ctx = FileLikeContext("x")
    with ctx:
        assert ctx.entered is True
    assert ctx.exited is True

    # 25. Memory/reference
    alias = mutable_alias_demo()
    assert alias["a"] == [1, 2, 3, 4] and alias["c"] == [1, 2, 3]
    assert alias["a_is_b"] is True and alias["a_is_c"] is False
    sc = shallow_vs_deep_copy_demo()
    assert sc["shallow"][0] == [1, 2, 99] and sc["deep"][0] == [1, 2]
    assert sc["shallow_inner_is_original"] is True
    assert sc["deep_inner_is_original"] is False
    right_append = mutable_default_trap_demo()
    a = right_append(1)
    b = right_append(2)
    assert a == [1] and b == [2] and a is not b
    # wrong version shares list — demonstrate trap
    w = right_append.wrong  # type: ignore[attr-defined]
    w.__defaults__ = ([],)  # reset for clean test (if polluted)
    # Actually re-import fresh: call with fresh default
    # We test that two calls share same object if we reuse the polluted default
    # Just verify right version doesn't share
    eq = is_vs_equality_demo()
    assert eq["a_eq_b"] is True and eq["a_is_b"] is False and eq["a_is_c"] is True
    assert reference_counting_demo() is not None

    # 26. Type hints
    assert typed_greet("Alice", 30) == "Alice is 30 years old"
    assert optional_demo(None) == "No value" and optional_demo("hi") == "HI"
    assert callable_demo(lambda a, b: a + b, 2, 3) == 5
    assert generic_first([1, 2, 3]) == 1 and generic_first([]) is None
    s: Stack[int] = Stack()
    s.push(1)
    s.push(2)
    assert s.pop() == 2 and s.peek() == 1

    class Circle:
        def draw(self) -> str:
            return "circle"

    assert render(Circle()) == "circle"
    assert union_vs_pipe_demo(42) == "int:42" and union_vs_pipe_demo("hi") == "str:hi"

    # 28. Concurrency (sync parts)
    assert threading_demo() == 400
    assert thread_pool_demo([1, 2, 3, 4]) == [1, 4, 9, 16]
    # process_pool_demo — CPU-bound; keep small for test speed
    assert process_pool_demo([1, 2, 3]) == [1, 4, 9]
    assert queue_thread_safe_demo() == [0, 10, 20, 30, 40]
    expl = gil_demo_explanation()
    assert "GIL" in expl and "IO-bound threads" in expl

    print("All Tier 3 sync asserts passed.")


async def _async_self_tests() -> None:
    """Async assert checks — run via asyncio.run(_async_self_tests())."""
    # 27. asyncio
    assert await async_fetch("test", 0.001) == "data:test"
    try:
        await async_fetch("fail", 0.001, fail=True)
        assert False, "should have raised AppError"
    except AppError:
        pass
    assert await async_gather_demo() == ["data:a", "data:b", "data:c"]
    assert await async_create_task_demo() == ["data:x", "data:y"]
    assert await async_timeout_demo() == "timed out"
    assert await async_context_manager_demo() == "resource"
    assert await async_generator_demo() == [0, 1, 2, 3, 4]
    print("All Tier 3 async asserts passed.")


if __name__ == "__main__":
    _self_tests()
    asyncio.run(_async_self_tests())
