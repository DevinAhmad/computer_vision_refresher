"""
Tier 1 — Must Know: Python Fundamentals
========================================

This module is IMPORT-ONLY (no side-effects on import).  All demonstrations
are provided as small, well-commented functions/classes plus a guarded
``_self_tests()`` function that you can run manually:

    import tier1_must_know
    tier1_must_know._self_tests()        # runs assert-based checks
    # or
    python tier1_must_know.py           # via __main__ guard

Topics covered
--------------
1.  Lists
2.  Dictionaries
3.  Sets
4.  Strings
5.  Functions (including *args, **kwargs, closures, lambdas)
6.  Loops & Comprehensions
7.  Sorting
8.  collections (Counter, defaultdict, deque, namedtuple, OrderedDict)
9.  heapq
10. Big-O (annotated complexities + timing helper)

Big-O Quick Reference (used in docstrings below)
-------------------------------------------------
O(1)       constant    — hash lookup, append, index access
O(log n)   logarithmic — binary search, heap push/pop
O(n)       linear      — single loop, Counter, heapify
O(n log n) linearithmic — sorting (Timsort), heap sort
O(n^2)     quadratic   — nested loops, naive duplicate check
O(2^n)     exponential — naive recursion without memo
"""

from __future__ import annotations

import heapq
import time
from collections import Counter, OrderedDict, defaultdict, deque, namedtuple
from typing import Any, Callable, Iterable

# =============================================================================
# 1. LISTS
# =============================================================================
# Lists are dynamic arrays: O(1) index access & append, O(n) insert/delete
# in the middle, O(k) slicing where k is slice size. They are mutable and
# preserve insertion order.


def deduplicate_preserve_order(nums: list[int]) -> list[int]:
    """
    Remove duplicates while preserving first-occurrence order.

    Approach: iterate once, track seen set.
    Complexity: Time O(n), Space O(n).
    """
    seen: set[int] = set()
    result: list[int] = []
    for x in nums:  # single pass O(n)
        if x not in seen:  # O(1) avg hash lookup
            seen.add(x)
            result.append(x)  # O(1) amortised
    return result


def flatten(nested: list[list[Any]]) -> list[Any]:
    """
    Flatten a 2-D list into 1-D.

    Complexity: Time O(N) where N = total elements, Space O(N).
    Uses a nested comprehension — very Pythonic.
    """
    # Outer loop over sublists, inner loop over elements
    return [elem for sublist in nested for elem in sublist]


def rotate(nums: list[int], k: int) -> list[int]:
    """
    Rotate list to the right by k steps WITHOUT mutating the original.

    Uses slicing: O(n) time and O(n) space (creates new list).
    Example: [1,2,3,4,5], k=2 -> [4,5,1,2,3]
    """
    if not nums:
        return []
    k = k % len(nums)  # handle k > len(nums)
    # Slicing creates new lists; concatenation is O(n)
    return nums[-k:] + nums[:-k]


def rotate_in_place(nums: list[int], k: int) -> None:
    """
    Rotate in-place using the 3-reverse trick (O(n) time, O(1) extra space).

    Steps: reverse whole array, reverse first k, reverse rest.
    This mutates the original list — often asked in interviews.
    """
    if not nums:
        return
    k %= len(nums)

    def reverse(lo: int, hi: int) -> None:
        while lo < hi:
            nums[lo], nums[hi] = nums[hi], nums[lo]
            lo += 1
            hi -= 1

    reverse(0, len(nums) - 1)
    reverse(0, k - 1)
    reverse(k, len(nums) - 1)


def chunk_list(nums: list[Any], size: int) -> list[list[Any]]:
    """
    Split list into chunks of given size. Last chunk may be smaller.
    Time O(n), Space O(n).
    """
    # Step through with stride `size`; slicing is O(size) each
    return [nums[i : i + size] for i in range(0, len(nums), size)]


# =============================================================================
# 2. DICTIONARIES
# =============================================================================
# Dicts are hash maps: avg O(1) get/set/delete, O(n) iteration.
# Since Python 3.7 they preserve insertion order.


def freq_count(words: list[str]) -> dict[str, int]:
    """
    Count word frequencies manually (without Counter) to show dict mechanics.

    Time O(n), Space O(u) where u = unique words.
    """
    freq: dict[str, int] = {}
    for w in words:
        # dict.get provides a default without KeyError — O(1)
        freq[w] = freq.get(w, 0) + 1
        # Alternative: freq[w] = freq.setdefault(w, 0) + 1  (but get is clearer)
    return freq


def merge_dicts(a: dict[Any, Any], b: dict[Any, Any]) -> dict[Any, Any]:
    """
    Merge two dicts; values from b overwrite a on key collision.

    Demonstrates three idioms:
      1) {**a, **b}  2) a | b  (Python 3.9+)  3) dict(a) + update
    Time O(n+m), Space O(n+m).
    """
    # Using the modern union operator (Python 3.9+) — most readable
    return a | b
    # Legacy alternative: return {**a, **b}


def group_by_length(words: list[str]) -> dict[int, list[str]]:
    """
    Group words by their length using defaultdict(list).

    Time O(n * L) where L is avg word length for len() — effectively O(n).
    """
    groups: defaultdict[int, list[str]] = defaultdict(list)
    for w in words:
        groups[len(w)].append(w)  # defaultdict auto-creates empty list
    return dict(groups)  # cast back to plain dict for clean return type


def invert_dict(d: dict[Any, Any]) -> dict[Any, list[Any]]:
    """
    Invert a dict: value -> list of keys that mapped to it.
    Useful when values are not unique.

    Time O(n), Space O(n).
    """
    inverted: defaultdict[Any, list[Any]] = defaultdict(list)
    for k, v in d.items():
        inverted[v].append(k)
    return dict(inverted)


# =============================================================================
# 3. SETS
# =============================================================================
# Sets are hash-based collections of unique, hashable elements.
# Avg O(1) add/lookup/discard, O(n) iteration, O(n+m) set operations.


def find_duplicates(nums: list[int]) -> set[int]:
    """
    Return set of values that appear more than once.

    Time O(n), Space O(n).
    """
    seen: set[int] = set()
    dups: set[int] = set()
    for x in nums:
        if x in seen:
            dups.add(x)
        else:
            seen.add(x)
    return dups


def set_operations_demo(a: set[int], b: set[int]) -> dict[str, set[int]]:
    """
    Showcase core set algebra with verbose comments.

    - Union (a | b): elements in either set
    - Intersection (a & b): elements in both
    - Difference (a - b): in a but not b
    - Symmetric difference (a ^ b): in exactly one of the two

    Each operation is O(len(a) + len(b)) avg case.
    """
    return {
        "union": a | b,
        "intersection": a & b,
        "difference_a_minus_b": a - b,
        "symmetric_difference": a ^ b,
        "is_subset": a <= b,  # type: ignore[dict-item]  # bool for demo
        "is_superset": a >= b,  # type: ignore[dict-item]
    }


def unique_intersection(a: list[int], b: list[int]) -> list[int]:
    """
    Intersection of two lists returning unique values (order not guaranteed).

    Time O(n+m) using set conversion. Use sorted() if deterministic order needed.
    """
    return list(set(a) & set(b))


# =============================================================================
# 4. STRINGS
# =============================================================================
# Strings are immutable sequences of Unicode code points.
# Slicing, concatenation (O(n)), and most methods return NEW strings.


def is_palindrome(s: str) -> bool:
    """
    Check if string is palindrome ignoring case and non-alphanumerics.

    Two-pointer technique on filtered string.
    Time O(n), Space O(n) for filtered copy (could be O(1) with in-place pointers).
    """
    # Filter to alphanumerics and lower-case — O(n)
    filtered = "".join(ch.lower() for ch in s if ch.isalnum())
    # Pythonic reversal check: filtered == filtered[::-1]
    # Explicit two-pointer for interview clarity:
    left, right = 0, len(filtered) - 1
    while left < right:
        if filtered[left] != filtered[right]:
            return False
        left += 1
        right -= 1
    return True


def is_anagram(a: str, b: str) -> bool:
    """
    Check if two strings are anagrams (same character counts).

    Counter-based: Time O(n), Space O(k) where k = alphabet size.
    Sorted comparison would be O(n log n).
    """
    # Quick length check — O(1) early exit
    if len(a) != len(b):
        return False
    return Counter(a) == Counter(b)


def longest_common_prefix(strs: list[str]) -> str:
    """
    Find longest common prefix among strings via vertical scanning.

    Time O(S) where S = sum of all chars, Space O(1) extra.
    """
    if not strs:
        return ""
    # Use first string as reference
    prefix = strs[0]
    for s in strs[1:]:
        # Shrink prefix until s starts with it
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def count_vowels(s: str) -> dict[str, int]:
    """
    Count vowels in a string (case-insensitive) — demonstrates string iteration.

    Time O(n), Space O(1) (fixed vowel set).
    """
    vowels = set("aeiou")
    counts: dict[str, int] = Counter(ch.lower() for ch in s if ch.lower() in vowels)
    return dict(counts)


# =============================================================================
# 5. FUNCTIONS
# =============================================================================
# Functions are first-class objects. Showcases *args, **kwargs, closures,
# lambdas, higher-order functions, and default-argument pitfalls.


def greet(name: str, greeting: str = "Hello") -> str:
    """
    Simple function with default argument. Default args are evaluated once at
    definition time — avoid mutable defaults!
    """
    return f"{greeting}, {name}!"


def flexible_sum(*args: int, **kwargs: int) -> int:
    """
    Demonstrate *args (tuple of positional) and **kwargs (dict of keyword).

    *args collects extra positional args, **kwargs collects extra keyword args.
    Time O(n) where n = total numbers passed.
    """
    total = sum(args) + sum(kwargs.values())
    return total


def make_adder(n: int) -> Callable[[int], int]:
    """
    Closure factory: returns a function that adds n to its argument.

    Closures capture variables from the enclosing scope (here `n`).
    Each call to make_adder creates a new closure with its own `n`.
    """

    def adder(x: int) -> int:
        # `n` is looked up in the enclosing scope (closure cell)
        return x + n

    return adder


def apply_twice(func: Callable[[Any], Any], value: Any) -> Any:
    """
    Higher-order function: applies func twice.

    Demonstrates functions as first-class values.
    Example: apply_twice(lambda x: x*2, 3) -> 12
    """
    return func(func(value))


def safe_append(item: Any, target: list[Any] | None = None) -> list[Any]:
    """
    Correct handling of mutable default argument.

    WRONG: def bad(item, target=[]):  # shared list across calls!
    RIGHT: use None sentinel and create new list inside.
    """
    if target is None:
        target = []
    target.append(item)
    return target


# Lambda example (kept as named function for testability)
square = lambda x: x * x  # noqa: E731  — intentional lambda demo


# =============================================================================
# 6. LOOPS & COMPREHENSIONS
# =============================================================================
# Comprehensions are syntactic sugar for loops that build collections.
# They are generally faster than explicit loops (C-level loop) and more readable.


def squares_comprehension(n: int) -> list[int]:
    """
    Return squares 0..n-1 via list comprehension.

    [x*x for x in range(n)]  vs  explicit loop — comp is ~20-30% faster.
    Time O(n), Space O(n).
    """
    return [x * x for x in range(n)]


def even_squares(n: int) -> list[int]:
    """
    List comp with condition: only even numbers squared.

    Demonstrates filtering inside comprehension.
    """
    return [x * x for x in range(n) if x % 2 == 0]


def dict_comprehension_demo(words: list[str]) -> dict[str, int]:
    """
    Dict comprehension: word -> length.
    Time O(n), Space O(n).
    """
    return {w: len(w) for w in words}


def set_comprehension_demo(nums: list[int]) -> set[int]:
    """
    Set comprehension: unique squares.
    """
    return {x * x for x in nums}


def nested_comprehension(matrix: list[list[int]]) -> list[int]:
    """
    Flatten matrix via nested comprehension (same as flatten() but via helper).
    Shows the order: [expr for row in matrix for val in row]
    Time O(N) where N total elements.
    """
    return [val for row in matrix for val in row]


def enumerate_zip_demo(names: list[str], scores: list[int]) -> list[str]:
    """
    Demonstrate enumerate + zip together.

    enumerate gives (index, value), zip pairs multiple iterables.
    Time O(n).
    """
    result: list[str] = []
    for idx, (name, score) in enumerate(zip(names, scores)):
        result.append(f"{idx}: {name}={score}")
    return result


# =============================================================================
# 7. SORTING
# =============================================================================
# Python uses Timsort: O(n log n) worst/average, O(n) best (already sorted),
# stable (preserves order of equal keys), in-place for list.sort().


def sort_by_key_demo(words: list[str]) -> list[str]:
    """
    Sort words by length, then alphabetically (stable sort trick).

    Demonstrates key= lambda/callable and stable sort property.
    Time O(n log n).
    """
    # Sort by length first; Python's sort is stable so alphabetical order
    # is preserved for equal lengths if we sort alphabetically first.
    return sorted(words, key=lambda w: (len(w), w))


def sort_tuples(pairs: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    Sort list of (name, age) tuples by age ascending, then name.
    Time O(n log n).
    """
    return sorted(pairs, key=lambda p: (p[1], p[0]))


def sort_with_comparator_demo(nums: list[int]) -> list[int]:
    """
    When key isn't enough, use functools.cmp_to_key to convert comparator.

    Here we sort by absolute value then by actual value for tie-break.
    In practice prefer key=abs; this is just to show cmp_to_key.
    """
    from functools import cmp_to_key

    def cmp(a: int, b: int) -> int:
        # Compare by abs value; if equal, smaller actual value first
        if abs(a) != abs(b):
            return abs(a) - abs(b)
        return a - b

    return sorted(nums, key=cmp_to_key(cmp))


def custom_object_sort_demo() -> list[tuple[str, int]]:
    """
    Sort list of dicts/objects by multiple criteria — common interview ask.

    Returns sorted list of (name, score) by score DESC then name ASC.
    """
    data = [("Alice", 90), ("Bob", 90), ("Charlie", 85)]
    # Negate score for descending, name stays ascending
    return sorted(data, key=lambda x: (-x[1], x[0]))


# =============================================================================
# 8. COLLECTIONS
# =============================================================================
# The collections module provides high-performance container datatypes.


def counter_demo(words: list[str]) -> dict[str, int]:
    """
    Counter counts hashable objects; most_common(k) is O(n log k) via heap.
    """
    c = Counter(words)  # O(n) counting
    # Most common 2 — demonstrates heap-backed method
    _top2 = c.most_common(2)  # noqa: F841 (demo only)
    # Counter supports arithmetic: c1 + c2, c1 - c2, etc.
    return dict(c)


def defaultdict_demo(pairs: list[tuple[str, int]]) -> dict[str, list[int]]:
    """
    defaultdict avoids KeyError by auto-creating default values.

    defaultdict(list) -> missing key creates []
    defaultdict(int)  -> missing key creates 0
    defaultdict(set)  -> missing key creates set()
    Time O(n).
    """
    dd: defaultdict[str, list[int]] = defaultdict(list)
    for key, val in pairs:
        dd[key].append(val)
    return dict(dd)


def deque_demo() -> list[int]:
    """
    deque is a double-ended queue: O(1) append/appendleft/popleft.

    List would be O(n) for popleft/insert(0). Ideal for BFS, sliding window.
    """
    dq: deque[int] = deque([1, 2, 3])
    dq.append(4)  # right — O(1)
    dq.appendleft(0)  # left — O(1)
    dq.popleft()  # remove from left — O(1)
    dq.rotate(1)  # rotate right by 1: [4,1,2,3] — O(k)
    return list(dq)


def namedtuple_demo() -> list[tuple]:
    """
    namedtuple creates lightweight, immutable record types with named fields.

    More readable than plain tuples, more memory-efficient than dicts.
    """
    Point = namedtuple("Point", ["x", "y"])
    p1 = Point(1, 2)
    p2 = Point(x=3, y=4)
    # Access by name or index: p1.x == p1[0]
    return [p1, p2, p1._asdict()]  # _asdict returns OrderedDict


def ordered_dict_demo() -> list[str]:
    """
    OrderedDict preserves insertion order and offers move_to_end, popitem.

    Since Python 3.7 plain dict also preserves order, but OrderedDict has
    extra methods and its equality checks order (dict equality ignores order).
    """
    od: OrderedDict[str, int] = OrderedDict()
    od["first"] = 1
    od["second"] = 2
    od["third"] = 3
    od.move_to_end("first")  # move to end — O(1)
    return list(od.keys())  # ['second', 'third', 'first']


# =============================================================================
# 9. HEAPQ
# =============================================================================
# heapq implements a min-heap via a list. heap[0] is always smallest.
# push/pop O(log n), heapify O(n), nsmallest/nlargest O(n log k).


def k_largest(nums: list[int], k: int) -> list[int]:
    """
    Return k largest elements using heapq.nlargest.

    Internally uses a min-heap of size k: Time O(n log k), Space O(k).
    If k ~ n, sorting O(n log n) may be similar; nlargest shines for small k.
    """
    if k <= 0:
        return []
    # heapq.nlargest is optimized and more readable than manual heap
    return heapq.nlargest(k, nums)


def k_smallest(nums: list[int], k: int) -> list[int]:
    """Return k smallest — O(n log k) via heapq.nsmallest."""
    if k <= 0:
        return []
    return heapq.nsmallest(k, nums)


def merge_sorted(lists: list[list[int]]) -> list[int]:
    """
    Merge multiple sorted lists into one sorted list using heapq.merge.

    heapq.merge is lazy (returns iterator) and runs in O(N log k) where
    N = total elements, k = number of lists. Much faster than concat+sort.
    """
    # heapq.merge assumes each input is already sorted
    return list(heapq.merge(*lists))


def heap_push_pop_demo(nums: list[int]) -> list[int]:
    """
    Manual heap operations to show heapq mechanics.

    Demonstrates heapify, heappush, heappop, heapreplace.
    """
    heap = nums[:]  # copy so we don't mutate input
    heapq.heapify(heap)  # O(n) — transform list into heap in-place
    heapq.heappush(heap, 0)  # O(log n)
    smallest = heapq.heappop(heap)  # O(log n) — should be 0 or min
    assert smallest == min(nums + [0])
    # heapreplace pops then pushes atomically — more efficient than pop+push
    if heap:
        heapq.heapreplace(heap, 100)  # O(log n)
    return heap


def max_heap_demo(nums: list[int]) -> list[int]:
    """
    Python has only min-heap; simulate max-heap by negating values.

    Common interview trick for "k largest" or "median finder" problems.
    """
    max_heap = [-x for x in nums]  # negate
    heapq.heapify(max_heap)  # min-heap of negatives == max-heap of originals
    # Pop largest (which is smallest negative)
    largest = -heapq.heappop(max_heap)  # O(log n)
    assert largest == max(nums)
    # Convert back for inspection
    return [-x for x in max_heap]


# =============================================================================
# 10. BIG-O — Annotated helper & comparison
# =============================================================================


def big_o_timing_decorator(func: Callable) -> Callable:
    """
    Decorator that prints execution time — helps empirically verify Big-O.

    Usage:
        @big_o_timing_decorator
        def my_func(n): ...

    Note: Single timing is noisy; for real benchmarks use timeit.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        # Verbose comment: we print rather than log for demo simplicity
        print(f"[Big-O timing] {func.__name__} took {elapsed:.6f}s")
        return result

    return wrapper


def contains_duplicate_naive(nums: list[int]) -> bool:
    """
    Naive O(n^2) duplicate check — nested loops.

    For each element, scan remaining elements. Correct but slow.
    Space O(1) extra.
    """
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return True
    return False


def contains_duplicate_optimized(nums: list[int]) -> bool:
    """
    Optimized O(n) duplicate check using a set.

    Trades O(n) space for O(n) time — classic space-time tradeoff.
    Space O(n).
    """
    seen: set[int] = set()
    for x in nums:
        if x in seen:  # O(1) avg
            return True
        seen.add(x)
    return False


def big_o_cheat_sheet() -> dict[str, str]:
    """
    Return a Big-O cheat-sheet dict for quick reference during interviews.

    Keys are operations, values are complexities with brief notes.
    """
    return {
        "list append": "O(1) amortised",
        "list insert(0)": "O(n) — shifts elements",
        "list index access": "O(1)",
        "list slice": "O(k) where k = slice size",
        "dict get/set": "O(1) avg, O(n) worst (hash collisions)",
        "set add/contains": "O(1) avg",
        "sorting (Timsort)": "O(n log n) avg/worst, O(n) best",
        "heap push/pop": "O(log n)",
        "heapify": "O(n)",
        "binary search": "O(log n)",
        "BFS/DFS graph": "O(V + E)",
        "naive duplicate check": "O(n^2) time, O(1) space",
        "set duplicate check": "O(n) time, O(n) space",
    }


# =============================================================================
# Self-tests (IMPORT-ONLY — not executed on import)
# =============================================================================


def _self_tests() -> None:
    """Run assert-based checks for every section. Call manually or via __main__."""
    # 1. Lists
    assert deduplicate_preserve_order([1, 2, 2, 3, 1]) == [1, 2, 3]
    assert flatten([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]
    assert flatten([]) == []
    assert rotate([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]
    assert rotate([], 3) == []
    assert rotate([1], 10) == [1]
    tmp = [1, 2, 3, 4, 5]
    rotate_in_place(tmp, 2)
    assert tmp == [4, 5, 1, 2, 3]
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    # 2. Dictionaries
    assert freq_count(["a", "b", "a"]) == {"a": 2, "b": 1}
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}
    assert group_by_length(["a", "bb", "ccc", "dd"]) == {1: ["a"], 2: ["bb", "dd"], 3: ["ccc"]}
    assert invert_dict({"a": 1, "b": 1, "c": 2}) == {1: ["a", "b"], 2: ["c"]}

    # 3. Sets
    assert find_duplicates([1, 2, 2, 3, 3, 3]) == {2, 3}
    assert find_duplicates([1, 2, 3]) == set()
    ops = set_operations_demo({1, 2, 3}, {3, 4, 5})
    assert ops["union"] == {1, 2, 3, 4, 5}
    assert ops["intersection"] == {3}
    assert ops["difference_a_minus_b"] == {1, 2}
    assert set(unique_intersection([1, 2, 2, 3], [2, 3, 4])) == {2, 3}

    # 4. Strings
    assert is_palindrome("A man, a plan, a canal: Panama") is True
    assert is_palindrome("race a car") is False
    assert is_palindrome("") is True
    assert is_anagram("listen", "silent") is True
    assert is_anagram("hello", "bello") is False
    assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
    assert longest_common_prefix([]) == ""
    assert longest_common_prefix(["alone"]) == "alone"
    assert count_vowels("Hello World") == {"e": 1, "o": 2}

    # 5. Functions
    assert greet("Alice") == "Hello, Alice!"
    assert greet("Bob", "Hi") == "Hi, Bob!"
    assert flexible_sum(1, 2, 3, a=4, b=5) == 15
    assert flexible_sum() == 0
    add5 = make_adder(5)
    add10 = make_adder(10)
    assert add5(3) == 8
    assert add10(3) == 13
    assert apply_twice(lambda x: x * 2, 3) == 12
    assert square(5) == 25
    # Mutable default pitfall check
    a = safe_append(1)
    b = safe_append(2)
    assert a == [1] and b == [2] and a is not b

    # 6. Loops/comprehensions
    assert squares_comprehension(5) == [0, 1, 4, 9, 16]
    assert even_squares(6) == [0, 4, 16]
    assert dict_comprehension_demo(["hi", "hello"]) == {"hi": 2, "hello": 5}
    assert set_comprehension_demo([1, 2, 2, 3]) == {1, 4, 9}
    assert nested_comprehension([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert enumerate_zip_demo(["Alice", "Bob"], [90, 85]) == ["0: Alice=90", "1: Bob=85"]

    # 7. Sorting
    assert sort_by_key_demo(["ccc", "a", "bb", "a"]) == ["a", "a", "bb", "ccc"]
    assert sort_tuples([("Bob", 30), ("Alice", 25), ("Charlie", 30)]) == [
        ("Alice", 25),
        ("Bob", 30),
        ("Charlie", 30),
    ]
    assert sort_with_comparator_demo([3, -2, -3, 1]) == [1, -2, -3, 3]
    assert custom_object_sort_demo() == [("Alice", 90), ("Bob", 90), ("Charlie", 85)]

    # 8. collections
    assert counter_demo(["a", "b", "a", "a"]) == {"a": 3, "b": 1}
    assert defaultdict_demo([("a", 1), ("a", 2), ("b", 3)]) == {"a": [1, 2], "b": [3]}
    assert deque_demo() == [4, 1, 2, 3]
    pts = namedtuple_demo()
    assert pts[0].x == 1 and pts[0].y == 2
    assert pts[2] == {"x": 1, "y": 2}
    assert ordered_dict_demo() == ["second", "third", "first"]

    # 9. heapq
    assert sorted(k_largest([1, 5, 3, 4, 2], 2)) == [4, 5]
    assert sorted(k_smallest([1, 5, 3, 4, 2], 2)) == [1, 2]
    assert merge_sorted([[1, 3, 5], [2, 4, 6], [0, 7]]) == [0, 1, 2, 3, 4, 5, 6, 7]
    # heap_push_pop_demo returns heap list — check it is still a valid heap
    h = heap_push_pop_demo([3, 1, 2])
    assert h[0] == min(h)  # heap property
    assert max_heap_demo([1, 5, 3]) is not None

    # 10. Big-O
    assert contains_duplicate_naive([1, 2, 3, 2]) is True
    assert contains_duplicate_naive([1, 2, 3]) is False
    assert contains_duplicate_optimized([1, 2, 3, 2]) is True
    assert contains_duplicate_optimized([1, 2, 3]) is False
    assert "sorting (Timsort)" in big_o_cheat_sheet()

    print("All Tier 1 asserts passed.")


if __name__ == "__main__":
    _self_tests()
