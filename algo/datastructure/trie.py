from collections import deque
from typing import Dict, List, Optional, Set


class Trie:
    def __init__(self, label=None):
        if label:
            self.label = label
        else:
            self.label = "#"
        self.children: List[Trie] = []

        self._child_index: Dict = {}
        self._reachable_labels: Optional[Set] = None

    def __str__(self):
        return str(self.label)

    def is_root(self):
        return isinstance(self.label, str)

    def has_child(self, label) -> bool:
        return label in self._child_index

    def is_leaf(self):
        return len(self.children) == 0

    def get_child(self, label) -> 'Trie':
        return self._child_index[label]

    def traverse(self, label):
        return self.get_child(label).children[0]

    def get_children(self):
        return self.children

    def add_child(self, trie: 'Trie'):
        self.children.append(trie)
        self._child_index[trie.label] = trie

        self._reachable_labels = None

    def get_children_containing_label(self, label):
        """Return child labels whose subtrie contains *label*."""
        return [
            child.label
            for child in self.children
            if child.contains(label)
        ]

    def _build_reachable_labels(self) -> Set:
        """Build the reachable label set iteratively (BFS) to avoid stack overflow."""
        labels: Set = set()
        stack = deque([self])
        while stack:
            node = stack.pop()
            labels.add(node.label)
            for child in node.children:
                stack.append(child)
        return labels

    def contains(self, label) -> bool:
        """Check if *label* exists anywhere in this subtrie (iterative, cached)."""
        if self._reachable_labels is None:
            self._reachable_labels = self._build_reachable_labels()
        return label in self._reachable_labels

    def get_locations(self) -> set[str]:
        """Collect all distinct locations in this subtrie (iterative)."""
        locations: set[str] = set()
        stack = deque([self])
        while stack:
            node = stack.pop()
            if hasattr(node.label, "location"):
                locations.add(node.label.location)
            for child in node.children:
                stack.append(child)
        return locations
