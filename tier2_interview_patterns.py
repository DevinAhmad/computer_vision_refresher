"""
Tier 2 — Coding Interview Patterns
====================================

IMPORT-ONLY module.  No code runs on import; run tests manually:

    import tier2_interview_patterns
    tier2_interview_patterns._self_tests()
    # or
    python tier2_interview_patterns.py

Patterns covered (numbered as per prompt 11-20)
----------------------------------------------
11. Two pointers
12. Sliding window
13. Binary search
14. Stack
15. Queue / BFS
16. DFS
17. Trees
18. Graphs
19. Recursion / Backtracking
20. Dynamic programming

Shared helpers (TreeNode, ListNode) are defined once at top so that
all patterns can reuse them. Every function is annotated with Big-O
in its docstring and includes verbose inline comments explaining the
“why” behind each step — not just the “how”.
"""

from __future__ import annotations

import bisect
import heapq
from collections import deque
from typing import Any, Optional

# =============================================================================
# Shared Data Structures
# =============================================================================


class ListNode:
    """Singly linked list node — used by two-pointer demos."""

    def __init__(self, val: int = 0, next: Optional[ListNode] = None):
        self.val = val
        self.next = next

    def __repr__(self) -> str:
        return f"ListNode({self.val})"


class TreeNode:
    """Binary tree node — used by DFS/BFS/Tree sections."""

    def __init__(
        self,
        val: int = 0,
        left: Optional[TreeNode] = None,
        right: Optional[TreeNode] = None,
    ):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"TreeNode({self.val})"


def build_linked_list(vals: list[int]) -> Optional[ListNode]:
    """Helper: build linked list from Python list. Time O(n)."""
    dummy = ListNode(0)
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next


def linked_to_list(head: Optional[ListNode]) -> list[int]:
    """Helper: linked list -> Python list. Time O(n)."""
    out: list[int] = []
    while head:
        out.append(head.val)
        head = head.next
    return out


def build_tree_from_level_order(vals: list[Optional[int]]) -> Optional[TreeNode]:
    """
    Build binary tree from level-order list (None = missing node).
    Example: [1,2,3,None,4] ->     1
                                  / \
                                 2   3
                                  \
                                   4
    Time O(n), Space O(n).
    """
    if not vals or vals[0] is None:
        return None
    root = TreeNode(vals[0])  # type: ignore[arg-type]
    q: deque[TreeNode] = deque([root])
    i = 1
    while q and i < len(vals):
        node = q.popleft()
        # Left child
        if i < len(vals) and vals[i] is not None:
            node.left = TreeNode(vals[i])  # type: ignore[arg-type]
            q.append(node.left)
        i += 1
        # Right child
        if i < len(vals) and vals[i] is not None:
            node.right = TreeNode(vals[i])  # type: ignore[arg-type]
            q.append(node.right)
        i += 1
    return root


# =============================================================================
# 11. TWO POINTERS
# =============================================================================
# Two pointers reduces O(n^2) nested loops to O(n) by moving pointers
# intelligently based on problem constraints (sorted input, palindrome, etc.).
# Variants: opposite ends (towards each other), same direction (fast/slow).


def two_sum_sorted(nums: list[int], target: int) -> list[int]:
    """
    Find two numbers in SORTED array that sum to target. Return their indices (0-based).

    Technique: opposite-direction pointers. If sum too small, move left forward
    (need larger); if too big, move right backward. Requires sorted input.
    Time O(n), Space O(1).
    """
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            return [left, right]
        elif s < target:
            left += 1  # need larger sum
        else:
            right -= 1  # need smaller sum
    return []  # no solution (problem may guarantee one)


def remove_duplicates_sorted(nums: list[int]) -> int:
    """
    Remove duplicates in-place from sorted array. Return new length.

    Technique: slow/fast pointers. `slow` tracks last unique position,
    `fast` scans ahead. Overwrites duplicates in-place.
    Time O(n), Space O(1). Classic “two pointers same direction”.
    """
    if not nums:
        return 0
    # slow = index of last unique element
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]  # overwrite next unique slot
    return slow + 1  # length is index+1


def container_with_most_water(height: list[int]) -> int:
    """
    LeetCode 11 — max area between two lines.

    Pointer at widest container, move the shorter wall inward (only way to
    potentially increase area — taller wall limits area).
    Time O(n), Space O(1).
    """
    left, right = 0, len(height) - 1
    max_area = 0
    while left < right:
        width = right - left
        h = min(height[left], height[right])
        max_area = max(max_area, width * h)
        # Move the pointer with smaller height — greedy choice
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_area


def has_cycle_floyd(head: Optional[ListNode]) -> bool:
    """
    Detect cycle in linked list via Floyd's fast/slow pointers.

    Slow moves 1 step, fast moves 2. If they meet, cycle exists.
    Time O(n), Space O(1) — better than hash-set O(n) space solution.
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next  # type: ignore[assignment]
        fast = fast.next.next
        if slow is fast:
            return True
    return False


# =============================================================================
# 12. SLIDING WINDOW
# =============================================================================
# Sliding window maintains a window [left, right] and slides it to find
# optimal subarray/substring satisfying a condition.
# Types: fixed-size (k) vs variable-size (expand/shrink).


def max_sum_subarray_k(nums: list[int], k: int) -> int:
    """
    Maximum sum of any contiguous subarray of size k — FIXED window.

    Instead of recomputing sum each time O(n*k), slide by subtracting
    the element leaving and adding the new one: O(n).
    Time O(n), Space O(1).
    """
    if k <= 0 or k > len(nums):
        raise ValueError("k must be in [1, len(nums)]")
    # Initial window sum
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for right in range(k, len(nums)):
        # Slide: remove leftmost of window, add new rightmost
        window_sum += nums[right] - nums[right - k]
        max_sum = max(max_sum, window_sum)
    return max_sum


def longest_substring_without_repeat(s: str) -> int:
    """
    Length of longest substring without repeating characters — VARIABLE window.

    Use dict to store last index of each char. When duplicate found inside
    current window, jump left pointer past previous occurrence.
    Time O(n), Space O(min(n, alphabet)).
    """
    char_index: dict[str, int] = {}  # char -> last seen index
    left = 0
    max_len = 0
    for right, ch in enumerate(s):
        # If ch seen inside current window, shrink from left
        if ch in char_index and char_index[ch] >= left:
            left = char_index[ch] + 1  # move past previous occurrence
        char_index[ch] = right
        max_len = max(max_len, right - left + 1)
    return max_len


def min_window_substring(s: str, t: str) -> str:
    """
    Minimum window in s that contains all chars of t (with frequency).

    Classic variable window with Counter-like dict + formed counter.
    Time O(|s| + |t|), Space O(|t|).
    Returns "" if no window.
    """
    from collections import Counter

    if not t:
        return ""
    need: dict[str, int] = Counter(t)  # required counts
    window: dict[str, int] = {}
    required = len(need)  # distinct chars needed
    formed = 0  # how many distinct chars meet required count
    left = 0
    best = (float("inf"), 0, 0)  # (length, left, right)

    for right, ch in enumerate(s):
        window[ch] = window.get(ch, 0) + 1
        if ch in need and window[ch] == need[ch]:
            formed += 1

        # Try to shrink window while still valid
        while left <= right and formed == required:
            # Update best if smaller
            if right - left + 1 < best[0]:
                best = (right - left + 1, left, right)
            # Remove left char from window
            left_ch = s[left]
            window[left_ch] -= 1
            if left_ch in need and window[left_ch] < need[left_ch]:
                formed -= 1
            left += 1

    if best[0] == float("inf"):
        return ""
    return s[best[1] : best[2] + 1]


# =============================================================================
# 13. BINARY SEARCH
# =============================================================================
# Binary search requires sorted input. Halves search space each step.
# Time O(log n), Space O(1) iterative, O(log n) recursive (call stack).


def binary_search_iterative(nums: list[int], target: int) -> int:
    """
    Standard iterative binary search. Return index or -1.

    Maintains [lo, hi] inclusive interval. Mid avoids overflow via
    lo + (hi-lo)//2 (relevant in other languages; Python ints unbounded).
    Time O(log n), Space O(1).
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2  # overflow-safe mid
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1  # target in right half
        else:
            hi = mid - 1  # target in left half
    return -1


def binary_search_recursive(nums: list[int], target: int) -> int:
    """
    Recursive variant — cleaner but uses O(log n) stack space.
    Time O(log n), Space O(log n).
    """

    def helper(lo: int, hi: int) -> int:
        if lo > hi:
            return -1
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            return helper(mid + 1, hi)
        else:
            return helper(lo, mid - 1)

    return helper(0, len(nums) - 1)


def search_insert_position(nums: list[int], target: int) -> int:
    """
    Find index to insert target to keep sorted order (bisect_left).

    Uses same loop but returns lo when not found — lo is insertion point.
    Time O(log n), Space O(1).
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return lo


def bisect_demo(nums: list[int], target: int) -> dict[str, Any]:
    """
    Showcase Python's bisect module which implements binary search in C.

    bisect_left / bisect_right + insort all O(log n) for search + O(n) for insert.
    """
    idx_left = bisect.bisect_left(nums, target)
    idx_right = bisect.bisect_right(nums, target)
    # insort would mutate; we demo on copy
    copy = nums[:]
    bisect.insort(copy, target)
    return {
        "bisect_left": idx_left,
        "bisect_right": idx_right,
        "insorted": copy,
    }


# =============================================================================
# 14. STACK
# =============================================================================
# Stack: LIFO. Python list works: append = push O(1), pop() = pop O(1).
# Used for: parentheses, monotonic stacks, DFS, undo, expression evaluation.


def is_valid_parentheses(s: str) -> bool:
    """
    Check balanced parentheses using a stack.

    Push opening brackets; on closing, pop and verify match.
    Time O(n), Space O(n) worst case (all openers).
    """
    mapping = {")": "(", "}": "{", "]": "["}
    stack: list[str] = []
    for ch in s:
        if ch in "({[":
            stack.append(ch)  # push opener
        elif ch in mapping:
            if not stack or stack.pop() != mapping[ch]:
                return False  # mismatch or stack empty
        # ignore other characters (optional)
    return not stack  # valid iff stack empty


def next_greater_element(nums: list[int]) -> list[int]:
    """
    For each element, find next greater element to the right — MONOTONIC STACK.

    Monotonic decreasing stack: pop while top < current, current is NGE for popped.
    Time O(n) — each element pushed/popped once, Space O(n).
    """
    n = len(nums)
    result = [-1] * n
    stack: list[int] = []  # stack of indices with decreasing values
    for i, val in enumerate(nums):
        # Pop indices whose NGE is current val
        while stack and nums[stack[-1]] < val:
            idx = stack.pop()
            result[idx] = val
        stack.append(i)
    return result


def daily_temperatures(temps: list[int]) -> list[int]:
    """
    LeetCode 739 — days until warmer temperature (monotonic stack variant).

    Similar to NGE but store days difference instead of value.
    Time O(n), Space O(n).
    """
    n = len(temps)
    answer = [0] * n
    stack: list[int] = []  # indices
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            prev = stack.pop()
            answer[prev] = i - prev
        stack.append(i)
    return answer


class MinStack:
    """
    Stack that supports getMin() in O(1) — classic interview design.

    Stores (value, current_min) pairs. Each push computes new min.
    All ops O(1), Space O(n).
    """

    def __init__(self) -> None:
        self._stack: list[tuple[int, int]] = []  # (val, min_so_far)

    def push(self, val: int) -> None:
        cur_min = val if not self._stack else min(val, self._stack[-1][1])
        self._stack.append((val, cur_min))

    def pop(self) -> None:
        if not self._stack:
            raise IndexError("pop from empty MinStack")
        self._stack.pop()

    def top(self) -> int:
        if not self._stack:
            raise IndexError("top from empty MinStack")
        return self._stack[-1][0]

    def get_min(self) -> int:
        if not self._stack:
            raise IndexError("get_min from empty MinStack")
        return self._stack[-1][1]


# =============================================================================
# 15. QUEUE / BFS
# =============================================================================
# Queue: FIFO. Use collections.deque for O(1) popleft; list.pop(0) is O(n).
# BFS uses queue to explore level-by-level — good for shortest path in unweighted graph.


def bfs_shortest_path_grid(grid: list[list[int]], start: tuple[int, int], end: tuple[int, int]) -> int:
    """
    BFS shortest path in binary grid (0=free, 1=wall). 4-directional moves.

    Returns steps count, or -1 if unreachable.
    Time O(R*C), Space O(R*C) for visited + queue.
    """
    if not grid or not grid[0]:
        return -1
    rows, cols = len(grid), len(grid[0])
    if grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1:
        return -1

    visited: set[tuple[int, int]] = {start}
    q: deque[tuple[int, int, int]] = deque([(start[0], start[1], 0)])  # (r,c,dist)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while q:
        r, c, d = q.popleft()  # O(1) with deque
        if (r, c) == end:
            return d
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == 0:
                visited.add((nr, nc))
                q.append((nr, nc, d + 1))
    return -1


def level_order_traversal(root: Optional[TreeNode]) -> list[list[int]]:
    """
    BFS level-order traversal of binary tree (LeetCode 102).

    Use queue, process level by level.
    Time O(n), Space O(w) where w = max width.
    """
    if not root:
        return []
    result: list[list[int]] = []
    q: deque[TreeNode] = deque([root])
    while q:
        level_size = len(q)
        level: list[int] = []
        for _ in range(level_size):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(level)
    return result


def rotting_oranges(grid: list[list[int]]) -> int:
    """
    LeetCode 994 — min minutes to rot all oranges. Multi-source BFS.

    0=empty, 1=fresh, 2=rotten. BFS from all rotten initially.
    Time O(R*C), Space O(R*C).
    """
    if not grid or not grid[0]:
        return -1
    rows, cols = len(grid), len(grid[0])
    q: deque[tuple[int, int, int]] = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                q.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh += 1
    if fresh == 0:
        return 0  # nothing to rot

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    max_minutes = 0
    while q:
        r, c, d = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2  # rot it
                fresh -= 1
                q.append((nr, nc, d + 1))
                max_minutes = max(max_minutes, d + 1)
    return max_minutes if fresh == 0 else -1


# =============================================================================
# 16. DFS
# =============================================================================
# DFS explores as deep as possible before backtracking.
# Recursive is natural; iterative uses explicit stack. Both O(V+E).


def dfs_recursive_graph(graph: dict[Any, list[Any]], start: Any) -> list[Any]:
    """
    Recursive DFS on adjacency-list graph.

    Time O(V+E), Space O(V) for visited + recursion stack.
    """
    visited: set[Any] = set()
    order: list[Any] = []

    def dfs(node: Any) -> None:
        visited.add(node)
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return order


def dfs_iterative_graph(graph: dict[Any, list[Any]], start: Any) -> list[Any]:
    """
    Iterative DFS with explicit stack — avoids recursion limits.

    Note: order differs from recursive due to stack LIFO; push reversed neighbors
    if you need identical order.
    Time O(V+E), Space O(V).
    """
    visited: set[Any] = set()
    stack: list[Any] = [start]
    order: list[Any] = []
    while stack:
        node = stack.pop()  # LIFO — deep first
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        # Push neighbors in reverse to preserve natural order if desired
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)
    return order


def flood_fill(image: list[list[int]], sr: int, sc: int, new_color: int) -> list[list[int]]:
    """
    LeetCode 733 — flood fill via DFS recursive.

    Fill connected region (4-directional) of same original color.
    Time O(R*C) worst case, Space O(R*C) recursion depth.
    """
    if not image or not image[0]:
        return image
    rows, cols = len(image), len(image[0])
    orig = image[sr][sc]
    if orig == new_color:
        return image  # nothing to do

    def dfs(r: int, c: int) -> None:
        if r < 0 or r >= rows or c < 0 or c >= cols or image[r][c] != orig:
            return
        image[r][c] = new_color
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    dfs(sr, sc)
    return image


def num_islands(grid: list[list[str]]) -> int:
    """
    LeetCode 200 — count islands (connected '1's, 4-directional).

    For each unvisited land cell, DFS to sink the island, increment count.
    Time O(R*C), Space O(R*C) recursion/visited.
    """
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r: int, c: int) -> None:
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != "1":
            return
        grid[r][c] = "0"  # sink — marks visited
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1
                dfs(r, c)
    return count


# =============================================================================
# 17. TREES
# =============================================================================
# Binary tree traversals, BST validation, and common operations.
# Recursive solutions are elegant; iterative uses stack.


def inorder_traversal(root: Optional[TreeNode]) -> list[int]:
    """
    In-order: left -> root -> right. For BST gives sorted order.
    Time O(n), Space O(h) where h = height (O(n) worst, O(log n) balanced).
    """
    result: list[int] = []

    def helper(node: Optional[TreeNode]) -> None:
        if not node:
            return
        helper(node.left)
        result.append(node.val)
        helper(node.right)

    helper(root)
    return result


def preorder_traversal(root: Optional[TreeNode]) -> list[int]:
    """Pre-order: root -> left -> right. Time O(n)."""
    result: list[int] = []

    def helper(node: Optional[TreeNode]) -> None:
        if not node:
            return
        result.append(node.val)
        helper(node.left)
        helper(node.right)

    helper(root)
    return result


def postorder_traversal(root: Optional[TreeNode]) -> list[int]:
    """Post-order: left -> right -> root. Time O(n)."""
    result: list[int] = []

    def helper(node: Optional[TreeNode]) -> None:
        if not node:
            return
        helper(node.left)
        helper(node.right)
        result.append(node.val)

    helper(root)
    return result


def max_depth(root: Optional[TreeNode]) -> int:
    """
    Maximum depth (height) of binary tree — recursive DFS.

    Depth of empty tree = 0, single node = 1.
    Time O(n), Space O(h).
    """
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    Validate Binary Search Tree property: left < root < right (strictly).

    Use bounds (low, high) propagated down. In-order approach also works
    but bounds method handles duplicates/edge cases cleanly.
    Time O(n), Space O(h).
    """
    INF = float("inf")

    def helper(node: Optional[TreeNode], low: float, high: float) -> bool:
        if not node:
            return True
        if not (low < node.val < high):
            return False
        return helper(node.left, low, node.val) and helper(node.right, node.val, high)

    return helper(root, -INF, INF)


def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Invert/mirror binary tree (swap left/right at every node).
    Time O(n), Space O(h) recursion.
    """
    if not root:
        return None
    root.left, root.right = root.right, root.left
    invert_tree(root.left)
    invert_tree(root.right)
    return root


# =============================================================================
# 18. GRAPHS
# =============================================================================
# Graph representations: adjacency list (sparse, O(V+E)) vs matrix (dense, O(V^2)).
# This module uses adjacency lists (dict[node -> list[neighbors]]).


def build_adj_list(edges: list[tuple[Any, Any]], directed: bool = False) -> dict[Any, list[Any]]:
    """
    Build adjacency list from edge list.

    Time O(E), Space O(V+E).
    """
    adj: dict[Any, list[Any]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        if not directed:
            adj.setdefault(v, []).append(u)
        else:
            adj.setdefault(v, [])  # ensure v appears as key even if no outgoing
    return adj


def bfs_graph(graph: dict[Any, list[Any]], start: Any) -> list[Any]:
    """
    BFS traversal order from start — finds shortest path in unweighted graph.

    Time O(V+E), Space O(V).
    """
    visited: set[Any] = {start}
    q: deque[Any] = deque([start])
    order: list[Any] = []
    while q:
        node = q.popleft()
        order.append(node)
        for nb in graph.get(node, []):
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return order


def has_cycle_undirected(graph: dict[Any, list[Any]]) -> bool:
    """
    Detect cycle in UNDIRECTED graph via DFS with parent tracking.

    If we visit a node already visited that isn't the parent, cycle exists.
    Time O(V+E), Space O(V).
    """
    visited: set[Any] = set()

    def dfs(node: Any, parent: Any) -> bool:
        visited.add(node)
        for nb in graph.get(node, []):
            if nb not in visited:
                if dfs(nb, node):
                    return True
            elif nb != parent:
                return True  # back edge to ancestor (not parent)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node, None):
                return True
    return False


def dijkstra(graph: dict[Any, list[tuple[Any, int]]], start: Any) -> dict[Any, int]:
    """
    Dijkstra's shortest path for weighted graph with non-negative weights.

    Graph format: {u: [(v, weight), ...]}
    Uses min-heap (heapq). Time O((V+E) log V), Space O(V).
    NOT for negative weights — use Bellman-Ford instead.
    """
    INF = float("inf")  # type: ignore[assignment]
    dist: dict[Any, int] = {node: INF for node in graph}
    dist[start] = 0
    heap: list[tuple[int, Any]] = [(0, start)]  # (distance, node)

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # stale entry
        for v, w in graph.get(u, []):
            # Ensure v in dist (sink nodes may not be keys)
            if v not in dist:
                dist[v] = INF
            if dist[u] + w < dist[v]:  # type: ignore[operator]
                dist[v] = dist[u] + w  # type: ignore[operator]
                heapq.heappush(heap, (dist[v], v))
    return dist


def topological_sort_dfs(graph: dict[Any, list[Any]]) -> list[Any]:
    """
    Topological sort via DFS post-order (Kahn's BFS is alternative).

    Only valid for DAG (directed acyclic graph). Raises if cycle detected.
    Time O(V+E), Space O(V).
    """
    visited: set[Any] = set()
    temp: set[Any] = set()  # recursion stack for cycle detection
    order: list[Any] = []

    def dfs(node: Any) -> None:
        if node in temp:
            raise ValueError("Graph has a cycle — topological sort impossible")
        if node in visited:
            return
        temp.add(node)
        for nb in graph.get(node, []):
            dfs(nb)
        temp.remove(node)
        visited.add(node)
        order.append(node)  # post-order

    for node in list(graph):
        if node not in visited:
            dfs(node)
    return order[::-1]  # reverse post-order


# =============================================================================
# 19. RECURSION / BACKTRACKING
# =============================================================================
# Backtracking = DFS over decision tree + undo (pop) choices.
# Template: choose -> explore -> unchoose. Often with pruning.


def permutations(nums: list[int]) -> list[list[int]]:
    """
    Generate all permutations of distinct numbers — classic backtracking.

    Time O(n! * n) — n! permutations each of length n to copy,
    Space O(n) recursion depth + O(n! * n) output.
    """
    result: list[list[int]] = []
    used = [False] * len(nums)
    path: list[int] = []

    def backtrack() -> None:
        if len(path) == len(nums):
            result.append(path[:])  # copy
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            # Choose
            used[i] = True
            path.append(nums[i])
            # Explore
            backtrack()
            # Unchoose (backtrack)
            path.pop()
            used[i] = False

    backtrack()
    return result


def subsets(nums: list[int]) -> list[list[int]]:
    """
    Generate all subsets (power set) — backtracking.

    Time O(2^n * n), Space O(n) depth + O(2^n * n) output.
    """
    result: list[list[int]] = []
    path: list[int] = []

    def backtrack(start: int) -> None:
        result.append(path[:])  # every node is a valid subset
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return result


def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    """
    Find all combinations where chosen numbers sum to target (reuse allowed).

    Time O(2^t) worst where t = target/min(candidates), Space O(t) depth.
    """
    result: list[list[int]] = []
    path: list[int] = []

    def backtrack(start: int, remaining: int) -> None:
        if remaining == 0:
            result.append(path[:])
            return
        if remaining < 0:
            return  # prune
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            # Not i+1 because we can reuse same element
            backtrack(i, remaining - candidates[i])
            path.pop()

    backtrack(0, target)
    return result


def solve_n_queens(n: int) -> list[list[str]]:
    """
    N-Queens — place n queens on n×n board with no mutual attacks.

    Uses sets for columns, diagonals (r-c) and anti-diagonals (r+c).
    Time O(n!) with pruning, Space O(n) + output.
    Returns board representations as list of strings.
    """
    result: list[list[str]] = []
    board: list[list[str]] = [["."] * n for _ in range(n)]
    cols: set[int] = set()
    diag1: set[int] = set()  # r - c
    diag2: set[int] = set()  # r + c

    def backtrack(r: int) -> None:
        if r == n:
            result.append(["".join(row) for row in board])
            return
        for c in range(n):
            if c in cols or (r - c) in diag1 or (r + c) in diag2:
                continue
            # Choose
            board[r][c] = "Q"
            cols.add(c)
            diag1.add(r - c)
            diag2.add(r + c)
            # Explore
            backtrack(r + 1)
            # Unchoose
            board[r][c] = "."
            cols.remove(c)
            diag1.remove(r - c)
            diag2.remove(r + c)

    backtrack(0)
    return result


# =============================================================================
# 20. DYNAMIC PROGRAMMING
# =============================================================================
# DP optimizations of recursion: memoization (top-down) or tabulation (bottom-up).
# Identify: overlapping subproblems + optimal substructure.


def fib_memo(n: int, memo: Optional[dict[int, int]] = None) -> int:
    """
    Fibonacci with memoization (top-down DP).

    Naive recursion is O(2^n); memo reduces to O(n) time, O(n) space.
    """
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


def fib_tabulation(n: int) -> int:
    """
    Fibonacci bottom-up tabulation — O(n) time, O(1) space (only keep last 2).

    Iteratively build from base cases. More space-efficient than memo.
    """
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def climbing_stairs(n: int) -> int:
    """
    LeetCode 70 — ways to climb n stairs taking 1 or 2 steps.

    Recurrence: dp[n] = dp[n-1] + dp[n-2] (Fibonacci!). O(n) time, O(1) space.
    """
    if n <= 2:
        return n
    a, b = 1, 2  # dp[1], dp[2]
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


def coin_change(coins: list[int], amount: int) -> int:
    """
    LeetCode 322 — min coins to make amount (or -1 if impossible).

    Bottom-up DP: dp[x] = min coins for amount x.
    Transition: dp[x] = min(dp[x-c] + 1) for all c in coins where x>=c.
    Time O(amount * len(coins)), Space O(amount).
    """
    INF = amount + 1  # larger than any possible answer
    dp = [INF] * (amount + 1)
    dp[0] = 0  # base: 0 coins for amount 0
    for x in range(1, amount + 1):
        for c in coins:
            if x >= c:
                # If we use coin c, we need dp[x-c] + 1 coins
                dp[x] = min(dp[x], dp[x - c] + 1)
    return dp[amount] if dp[amount] != INF else -1


def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    LCS length — classic 2D DP.

    dp[i][j] = LCS of text1[:i] and text2[:j].
    If chars match: 1 + dp[i-1][j-1]; else max(dp[i-1][j], dp[i][j-1]).
    Time O(m*n), Space O(m*n) — can be optimized to O(min(m,n)).
    """
    m, n = len(text1), len(text2)
    # dp with extra row/col for empty prefix base case
    dp: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    """
    0/1 Knapsack — max value without exceeding capacity.

    Each item either taken or not (0/1). Bottom-up DP:
    dp[w] = max value achievable with capacity w.
    Iterate items outer, capacity inner in REVERSE to avoid reuse in same iteration.
    Time O(n * capacity), Space O(capacity).
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        # Reverse so each item used at most once
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]


# =============================================================================
# Self-tests (IMPORT-ONLY)
# =============================================================================


def _self_tests() -> None:
    """Run assert-based checks for every pattern."""
    # 11. Two pointers
    assert two_sum_sorted([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum_sorted([1, 2, 3], 7) == []
    nums = [0, 0, 1, 1, 1, 2, 2, 3]
    k = remove_duplicates_sorted(nums)
    assert nums[:k] == [0, 1, 2, 3] and k == 4
    assert container_with_most_water([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    # cycle
    head = build_linked_list([1, 2, 3])
    assert has_cycle_floyd(head) is False
    cyc = build_linked_list([1, 2, 3])
    assert cyc is not None
    # create cycle: tail -> head
    tail = cyc
    while tail.next:
        tail = tail.next
    tail.next = cyc
    assert has_cycle_floyd(cyc) is True
    tail.next = None  # break cycle for GC

    # 12. Sliding window
    assert max_sum_subarray_k([2, 1, 5, 1, 3, 2], 3) == 9  # [5,1,3]
    assert longest_substring_without_repeat("abcabcbb") == 3  # "abc"
    assert longest_substring_without_repeat("bbbbb") == 1
    assert longest_substring_without_repeat("") == 0
    assert min_window_substring("ADOBECODEBANC", "ABC") == "BANC"
    assert min_window_substring("a", "b") == ""

    # 13. Binary search
    assert binary_search_iterative([1, 3, 5, 7, 9], 5) == 2
    assert binary_search_iterative([1, 3, 5], 2) == -1
    assert binary_search_recursive([1, 3, 5, 7, 9], 9) == 4
    assert binary_search_recursive([], 1) == -1
    assert search_insert_position([1, 3, 5, 6], 5) == 2
    assert search_insert_position([1, 3, 5, 6], 2) == 1
    assert search_insert_position([1, 3, 5, 6], 7) == 4
    bdemo = bisect_demo([1, 2, 4, 4, 5], 4)
    assert bdemo["bisect_left"] == 2 and bdemo["bisect_right"] == 4

    # 14. Stack
    assert is_valid_parentheses("()[]{}") is True
    assert is_valid_parentheses("(]") is False
    assert is_valid_parentheses("([)]") is False
    assert is_valid_parentheses("") is True
    assert next_greater_element([2, 1, 2, 4, 3]) == [4, 2, 4, -1, -1]
    assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    ms = MinStack()
    ms.push(3)
    ms.push(5)
    assert ms.get_min() == 3
    ms.push(2)
    ms.push(1)
    assert ms.get_min() == 1
    ms.pop()
    assert ms.get_min() == 2

    # 15. Queue/BFS
    grid = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]
    assert bfs_shortest_path_grid(grid, (0, 0), (2, 2)) == 4
    assert bfs_shortest_path_grid([[1]], (0, 0), (0, 0)) == -1
    root = build_tree_from_level_order([3, 9, 20, None, None, 15, 7])
    assert level_order_traversal(root) == [[3], [9, 20], [15, 7]]
    assert level_order_traversal(None) == []
    assert rotting_oranges([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
    assert rotting_oranges([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1

    # 16. DFS
    graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
    assert set(dfs_recursive_graph(graph, "A")) == {"A", "B", "C", "D"}
    assert set(dfs_iterative_graph(graph, "A")) == {"A", "B", "C", "D"}
    img = [[1, 1, 1], [1, 1, 0], [1, 0, 1]]
    assert flood_fill([row[:] for row in img], 1, 1, 2) == [[2, 2, 2], [2, 2, 0], [2, 0, 1]]
    # num_islands mutates grid, so pass copy
    assert num_islands([list(row) for row in ["11000", "11000", "00100", "00011"]]) == 3
    assert num_islands([]) == 0

    # 17. Trees
    bst = build_tree_from_level_order([4, 2, 6, 1, 3, 5, 7])
    assert inorder_traversal(bst) == [1, 2, 3, 4, 5, 6, 7]
    assert preorder_traversal(bst) == [4, 2, 1, 3, 6, 5, 7]
    assert postorder_traversal(bst) == [1, 3, 2, 5, 7, 6, 4]
    assert max_depth(bst) == 3
    assert max_depth(None) == 0
    assert is_valid_bst(bst) is True
    bad = build_tree_from_level_order([5, 1, 4, None, None, 3, 6])
    assert is_valid_bst(bad) is False
    inv = build_tree_from_level_order([4, 2, 7, 1, 3, 6, 9])
    invert_tree(inv)
    assert level_order_traversal(inv) == [[4], [7, 2], [9, 6, 3, 1]]

    # 18. Graphs
    edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
    adj = build_adj_list(edges)
    assert set(adj[0]) == {1, 2}
    assert set(bfs_graph(adj, 0)) == {0, 1, 2, 3}
    assert has_cycle_undirected(adj) is True  # triangle 0-1-2
    tree_edges = {0: [1, 2], 1: [0], 2: [0]}
    assert has_cycle_undirected(tree_edges) is False
    # Dijkstra
    wgraph: dict[Any, list[tuple[Any, int]]] = {
        "A": [("B", 1), ("C", 4)],
        "B": [("C", 2), ("D", 5)],
        "C": [("D", 1)],
        "D": [],
    }
    dist = dijkstra(wgraph, "A")
    assert dist["A"] == 0 and dist["B"] == 1 and dist["C"] == 3 and dist["D"] == 4
    # Topological sort
    dag = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
    topo = topological_sort_dfs(dag)
    assert topo.index("A") < topo.index("B")
    assert topo.index("A") < topo.index("C")
    assert topo.index("D") > topo.index("B") and topo.index("D") > topo.index("C")

    # 19. Recursion/backtracking
    perms = permutations([1, 2, 3])
    assert len(perms) == 6 and [1, 2, 3] in perms
    # subsets: 2^n
    subs = subsets([1, 2])
    assert len(subs) == 4 and [] in subs and [1, 2] in subs
    cs = combination_sum([2, 3, 6, 7], 7)
    # order may vary, check sets
    assert any(sorted(c) == [2, 2, 3] for c in cs) and any(c == [7] for c in cs)
    q1 = solve_n_queens(1)
    assert q1 == [["Q"]]
    q4 = solve_n_queens(4)
    assert len(q4) == 2  # known: 4-queens has 2 solutions

    # 20. DP
    assert fib_memo(10) == 55
    assert fib_tabulation(10) == 55
    assert fib_memo(0) == 0 and fib_tabulation(1) == 1
    assert climbing_stairs(2) == 2
    assert climbing_stairs(3) == 3
    assert climbing_stairs(5) == 8
    assert coin_change([1, 3, 4], 6) == 2  # 3+3 or 2*? actually 3+3 =2 coins
    assert coin_change([2], 3) == -1
    assert longest_common_subsequence("abcde", "ace") == 3
    assert longest_common_subsequence("abc", "def") == 0
    assert knapsack_01([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9  # items 3+4 (cap7: 4+5? check)
    # weights [1,3,4,5] values [1,4,5,7] cap 7 -> best is 3+4 weight7? value 4+5=9
    assert knapsack_01([1, 2, 3], [10, 15, 40], 6) == 65  # all

    print("All Tier 2 asserts passed.")


if __name__ == "__main__":
    _self_tests()
