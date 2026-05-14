# Changes — Bug Fixes in Distributed Conformance Checking

This document catalogues each bug found and the corresponding fix applied.

---

## 1. Activity Naming — Occurrence Counters

**File:** `algo/utility/converter.py` → `to_event_location_aware()`

**Problem:**  
Activity names were constructed with embedded occurrence counters and location:  
`CRP-B0`, `CRP-B1`, etc. This inflated the activity alphabet from `|A|` to  
`|A| × |N| × ℕ`, making the trie hyper-specific to the training data.  
A validation trace with activities in a different order or count would never  
match any trie path, producing maximum-cost alignments (all log moves).

**Old code:**
```python
def to_event_location_aware(self, pm4py_event, case_id, location_key=None, occurrence=""):
    location = pm4py_event[location_key] if location_key and location_key in pm4py_event else ""
    return Event(
        time=pm4py_event["time:timestamp"],
        activity=f"{pm4py_event['concept:name']}-{location}{occurrence}",
        case_id=case_id,
        location=location
    )
```

**New code:**
```python
def to_event_location_aware(self, pm4py_event, case_id, location_key=None, occurrence=""):
    location = pm4py_event[location_key] if location_key and location_key in pm4py_event else ""
    return Event(
        time=pm4py_event["time:timestamp"],
        activity=pm4py_event["concept:name"],
        case_id=case_id,
        location=location
    )
```

---

## 2. Located Activity Equality — Location Ignored

**File:** `algo/datastructure/located_activity.py`

**Problem:**  
`__eq__` compared only the activity name, ignoring the location component.  
`__hash__` was based only on the activity name.  
This caused trie lookups to match activities at the wrong node — e.g.  
`CRP@B` would equal `CRP@A`, producing incorrect alignments.

**Old code:**
```python
def __eq__(self, other):
    if not isinstance(other, LocatedActivity):
        return False
    return self.activity == other.activity

def __hash__(self):
    return hash(self.activity)
```

**New code:**
```python
def __eq__(self, other):
    if not isinstance(other, LocatedActivity):
        return False
    return self.activity == other.activity and self.location == other.location

def same_activity(self, other):
    """Compare only the activity name, ignoring location."""
    if not isinstance(other, LocatedActivity):
        return False
    return self.activity == other.activity

def __hash__(self):
    return hash((self.activity, self.location))
```

---

## 3. Trie Containment — Recursive Stack Overflow

**File:** `algo/datastructure/trie.py`

**Problem:**  
`contains()` performed a deep recursive traversal on every call.  
With large tries (50+ training traces, 22+ locations), this exceeded  
Python's default recursion limit (1000), crashing with `RecursionError`.  
There was also no child index — finding a child required O(n) linear scan.

**Old code:**
```python
def contains(self, label) -> bool:
    if self.label == label:
        return True
    for child in self.children:
        if child.contains(label):
            return True
    return False

# No _child_index; get_child did linear scan:
def get_child(self, label) -> 'Trie':
    for child in self.children:
        if child.label == label:
            return child
    return None
```

**New code:**
```python
def __init__(self, label=None):
    # ...
    self._child_index: Dict = {}
    self._reachable_labels: Optional[Set] = None

def has_child(self, label) -> bool:
    return label in self._child_index

def get_child(self, label) -> 'Trie':
    return self._child_index[label]

def add_child(self, trie: 'Trie'):
    self.children.append(trie)
    self._child_index[trie.label] = trie
    self._reachable_labels = None  # invalidate cache

def _build_reachable_labels(self) -> Set:
    """Build the reachable label set iteratively (BFS)."""
    labels: Set = set()
    stack = deque([self])
    while stack:
        node = stack.pop()
        labels.add(node.label)
        for child in node.children:
            stack.append(child)
    return labels

def contains(self, label) -> bool:
    """Iterative, cached containment check."""
    if self._reachable_labels is None:
        self._reachable_labels = self._build_reachable_labels()
    return label in self._reachable_labels
```

---

## 4. Alignment Construction — Deep Copy Explosion

**File:** `algo/datastructure/alignment.py`

**Problem:**  
Every move operation (`sync_move`, `move_on_model_skip_log`,  
`move_on_log_skip_model`) called `copy.deepcopy(self)` on the entire  
alignment before appending the new element.  
For a trace of length *m* with A\* branching factor *b*, this produced  
O(b^m × m) copied elements — exponential memory and runtime.

**Old code:**
```python
import copy

def sync_move(self, sync_move):
    new_alignment = copy.deepcopy(self)
    new_alignment.elements.append(AlignmentElement(sync_move, sync_move))
    return new_alignment

def move_on_model_skip_log(self, model_move):
    new_alignment = copy.deepcopy(self)
    new_alignment.elements.append(AlignmentElement(model_move, SKIP))
    new_alignment._cost += MODEL_COST
    return new_alignment
```

**New code:**
```python
__slots__ = ('elements', '_cost', 'processed_events')

def _extend(self, element: AlignmentElement, extra_cost: int) -> 'Alignment':
    new_elements = list(self.elements)  # shallow copy — elements are immutable
    new_elements.append(element)
    return Alignment(new_elements, self._cost + extra_cost)

def sync_move(self, sync_move: LocatedActivity) -> 'Alignment':
    return self._extend(AlignmentElement(sync_move, sync_move), 0)

def move_on_model_skip_log(self, model_move: LocatedActivity) -> 'Alignment':
    return self._extend(AlignmentElement(model_move, SKIP), MODEL_COST)

def move_on_log_skip_model(self, log_move: LocatedActivity) -> 'Alignment':
    return self._extend(AlignmentElement(SKIP, log_move), LOG_COST)
```

---

## 5. Cycle Detection — Unbounded Mutual Recursion

**File:** `algo/alignment_node.py` → `find_best_alignment()`, `_request_external_alignment()`

**Problem:**  
If node A had an entry point labelled with location B, and node B had  
an entry point labelled with location A, the GetAlign protocol recursed  
indefinitely: A → B → A → B → …, producing a `RecursionError`.

**Old code:**
```python
def find_best_alignment(self, target=None, i=sys.maxsize, is_start=False):
    # No cycle detection at all
    for entry_point in self.model.get_children_containing_label(target):
        response, model = self._request_external_alignment(entry_point, i)
        # ...

def _request_external_alignment(self, entry_point, i):
    # Always recurses, no guard
    alignment_response = (
        self.network.get_node(entry_point.location)
        .get_alignment(entry_point, i)
    )
```

**New code:**
```python
def find_best_alignment(self, target=None, i=sys.maxsize, is_start=False,
                         _visiting: Set[str] | None = None):
    if _visiting is None:
        _visiting = set()
    _visiting = _visiting | {self.node_id}

    for entry_point in self.model.get_children_containing_label(target):
        response, model = self._request_external_alignment(
            entry_point, i, _visiting
        )
        # ...

def _request_external_alignment(self, entry_point, i, _visiting):
    # Guard: skip if already visiting this node
    if entry_point.location in _visiting:
        return None, model
    # Otherwise recurse normally, passing the visiting set
    alignment_response = (
        self.network.get_node(entry_point.location)
        .get_alignment(entry_point, i, _visiting)
    )
```

---

## 6. Cache — Stale Results Across Events

**File:** `algo/alignment_node.py` → `process_event()`, `_request_external_alignment()`

**Problem:**  
The cache was keyed only by entry-point label, without including the  
timestamp. Once an upstream response was cached for entry point *e*,  
all subsequent events reused that stale response regardless of how  
many new upstream events had arrived. The cache was also never cleared.

**Old code:**
```python
def __init__(self, ...):
    self.cache = {}

def _request_external_alignment(self, entry_point, i):
    cache_key = entry_point  # label only — ignores timestamp
    if cache_key in self.cache:
        alignment_response = self.cache[cache_key]
    else:
        alignment_response = ...
        self.cache[cache_key] = alignment_response
```

**New code:**
```python
def process_event(self, located_activity, i):
    self.i = i
    self.observed_events[i] = located_activity
    self.cache = {}  # clear per event
    # ...

def _request_external_alignment(self, entry_point, i, _visiting):
    cache_key = (entry_point, i)  # includes timestamp upper bound
    if cache_key in self.cache:
        alignment_response = self.cache[cache_key]
    else:
        alignment_response = ...
        if alignment_response is not None:
            self.cache[cache_key] = alignment_response
```

---

## 7. External Log Moves — Duplicates from Nested Addition

**File:** `algo/alignment_node.py` → `process_event()`, `find_best_alignment()`

**Problem:**  
`_add_external_log_moves` was called inside `find_best_alignment()`  
(the recursive function), not just at the top level.  
When node A composed an alignment from B, A added B's events as log moves.  
When B had composed from A, B had already added A's events as log moves.  
The result contained duplicate log moves, inflating the cost.

**Old code:**
```python
def find_best_alignment(self, ...):
    # ...
    # External log moves added here — inside the recursive function
    all_candidate_alignments = self._add_external_log_moves(
        all_candidate_alignments, i
    )
    best = min(all_candidate_alignments, key=...)
    return best
```

**New code:**
```python
def process_event(self, located_activity, i):
    # ...
    response = self.find_best_alignment(located_activity, i, is_start=True)
    # External log moves added ONLY here — at the top level
    responses = self._add_external_log_moves([response], i)
    return responses[0].alignment

def find_best_alignment(self, ...):
    # No _add_external_log_moves call inside here
    best = min(all_candidate_alignments, key=...)
    return best
```

---

## 8. Local Trace — Events Dropped by Upstream Timestamp Filter

**File:** `algo/alignment_node.py` → `_get_relevant_local_trace()`

**Problem:**  
The local trace was filtered to include only events *after* the upstream's  
last timestamp: `self._get_trace(response.timestamp, max_time)`.  
If local events occurred before the upstream transition, they were  
silently dropped, producing incomplete alignments.

**Old code:**
```python
def _get_relevant_local_trace(self, i, is_start, last_node, timestamp):
    max_time = i if not is_start else sys.maxsize
    return self._get_trace(timestamp, max_time)
    #                     ^^^^^^^^^
    # Only events after the upstream timestamp — drops early local events
```

**New code:**
```python
def _get_relevant_local_trace(self, i, is_start, last_node, timestamp):
    max_time = i if not is_start else sys.maxsize
    return self._get_trace(-1, max_time)
    #                      ^^
    # ALL local events up to current time — upstream timestamp is irrelevant
```

---

## 9. Coverage Check — One-Sided (Missing Model-Side Events)

**File:** `algo/datastructure/alignment.py` → `append_missing_log_moves()`

**Problem:**  
When adding external log moves, the coverage check only looked at the  
*log side* of existing alignment elements. Events that appeared on the  
*model side* of synchronous or model moves were not detected as covered,  
causing them to be re-added as duplicate log moves.

**Old code:**
```python
def append_missing_log_moves(self, log_moves):
    covered = set()
    for e in self.elements:
        if e.log != SKIP:
            covered.add(e.log)
        # Model side NOT checked
    result = self
    for log_move in log_moves:
        if log_move not in covered:
            result = result.move_on_log_skip_model(log_move)
    return result
```

**New code:**
```python
def append_missing_log_moves(self, log_moves):
    covered = set()
    for e in self.elements:
        if e.log != SKIP:
            covered.add(e.log)
        if e.model != SKIP:
            covered.add(e.model)     # ← also check model side
    result = self
    for log_move in log_moves:
        if log_move not in covered:
            result = result.move_on_log_skip_model(log_move)
            covered.add(log_move)    # ← prevent re-addition within this call
    return result
```

---

## 10. A\* Alignment Calculator — Robustness Fixes

**File:** `algo/alignments/alignment_calculator.py`

**Problem (a):**  
The A\* priority queue used `(Alignment, node, idx)` tuples. When two  
alignments had equal cost, Python attempted to compare `Trie` objects,  
raising `TypeError`.

**Problem (b):**  
The goal check compared only `current_node.label.activity == target.activity`,  
ignoring the location. This could cause premature termination at a wrong  
trie node.

**Problem (c):**  
If the search space was exhausted without finding a goal, the function  
returned `None`, causing `NoneType` errors downstream.

**Old code:**
```python
queue = [(Alignment(), start_node, 0)]

# Goal check — activity only
if current_node.label == target.activity:
    return path

# No fallback — returns None implicitly
```

**New code:**
```python
counter = 0
queue = [(Alignment(), counter, start_node, 0)]

# Goal check — full LocatedActivity equality
if (not current_node.is_root()
        and hasattr(current_node.label, 'activity')
        and current_node.label == target  # checks activity AND location
        and trace_idx == len(trace)):
    return path

# Fallback — never returns None
result = Alignment()
for event in trace:
    result = result.move_on_log_skip_model(event)
return result
```
