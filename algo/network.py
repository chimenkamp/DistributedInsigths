from __future__ import annotations
from typing import Dict, List

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Protocol, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


class Network[T]:

    def __init__(self):
        self.nodes: Dict[str, T] = {}

    def add_node(self, node_id, node):
        self.nodes[node_id] = node

    def get_node(self, node_id) -> T:
        return self.nodes[node_id]

    def get_all_nodes(self, own_node_id: str)-> List[T]:
        return [node for node in list(self.nodes.values()) if node.node_id != own_node_id]
    