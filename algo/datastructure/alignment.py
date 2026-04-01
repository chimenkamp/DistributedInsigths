from typing import List

from algo.datastructure.alignment_constants import SKIP, MODEL_COST, LOG_COST
from algo.datastructure.alignment_element import AlignmentElement
from algo.datastructure.located_activity import LocatedActivity


class Alignment:
    """Lightweight alignment that avoids deepcopy.

    New moves create a *new* Alignment with a shallow-copied element list
    plus one appended element.  AlignmentElement is immutable so sharing
    is safe.
    """

    __slots__ = ('elements', '_cost', 'processed_events')

    def __init__(self, elements: List[AlignmentElement] | None = None, cost: int = 0):
        self.elements: List[AlignmentElement] = elements if elements is not None else []
        self._cost: int = cost
        self.processed_events = 0


    def _extend(self, element: AlignmentElement, extra_cost: int) -> 'Alignment':
        new_elements = list(self.elements) 
        new_elements.append(element)
        return Alignment(new_elements, self._cost + extra_cost)

    def sync_move(self, sync_move: LocatedActivity) -> 'Alignment':
        return self._extend(AlignmentElement(sync_move, sync_move), 0)

    def move_on_model_skip_log(self, model_move: LocatedActivity) -> 'Alignment':
        return self._extend(AlignmentElement(model_move, SKIP), MODEL_COST)

    def move_on_log_skip_model(self, log_move: LocatedActivity) -> 'Alignment':
        return self._extend(AlignmentElement(SKIP, log_move), LOG_COST)

    def increment_processed_events(self):
        self.processed_events += 1

    def get_cost(self) -> int:
        return self._cost

    def is_empty(self):
        return not bool(self.elements)

    def get_all_log_moves(self):
        return [e.log for e in self.elements if e.log != SKIP]

    def contains_log_moves(self):
        return any(e.log != SKIP for e in self.elements)

    def __add__(self, other):
        if not other:
            return Alignment(list(self.elements), self._cost)
        new_elements = list(self.elements) + list(other.elements)
        return Alignment(new_elements, self._cost + other._cost)

    def __lt__(self, other):
        if self._cost != other._cost:
            return self._cost < other._cost
        if len(self.elements) != len(other.elements):
            return len(self.elements) < len(other.elements)
        return id(self) < id(other)

    def __eq__(self, other):
        if not isinstance(other, Alignment):
            return NotImplemented
        return self.elements == other.elements

    def __hash__(self):
        return hash(tuple(self.elements))

    def __str__(self):
        final_string = ""
        for element in self.elements:
            final_string += str(element) + "\n"
        return f"[Cost:{self.get_cost()}]\n{final_string}"


    def append_missing_log_moves(self, log_moves):
        """Append log moves for events not already covered by this alignment.

        An event is considered covered if it appears on either the log side
        (sync or log move) or model side (sync or model move) of any existing
        alignment element.  Uses counts so that duplicate activities (e.g.
        CRP@B occurring 5 times) are handled correctly.
        """
        from collections import Counter
        covered = Counter()
        for e in self.elements:
            if e.log != SKIP:
                covered[e.log] += 1
            if e.model != SKIP:
                covered[e.model] += 1
        result = self
        for log_move in log_moves:
            if covered[log_move] > 0:
                covered[log_move] -= 1
            else:
                result = result.move_on_log_skip_model(log_move)
        return result
