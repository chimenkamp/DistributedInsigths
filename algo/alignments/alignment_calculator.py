import heapq

from algo.datastructure.alignment import Alignment
from algo.datastructure.alignment_constants import SKIP
from algo.datastructure.located_activity import LocatedActivity
from algo.datastructure.trie import Trie


def _check_predecessors(path, required_predecessors, target) -> bool:
    """Verify that the model-side path through the target's location ends
    with the required predecessor sequence before the target.
    
    Uses ALL model-side moves (both sync and model moves) at the target
    location to determine which trie branch was traversed."""
    model_at_location = []
    for elem in path.elements:
        if elem.model != SKIP:
            if (hasattr(elem.model, 'location')
                    and elem.model.location == target.location):
                model_at_location.append(elem.model)

    if model_at_location and model_at_location[-1] == target:
        model_at_location = model_at_location[:-1]
    k = len(required_predecessors)
    available = len(model_at_location)
    if available == 0:
        return True  # no context to contradict
    compare_len = min(k, available)
    actual_tail = model_at_location[-compare_len:]
    expected_tail = required_predecessors[-compare_len:]
    for actual, expected in zip(actual_tail, expected_tail):
        if not actual.same_activity(expected):
            return False
    return True


def calculate_alignment(trace, trie_node: Trie, target=None, partial=False,
                        required_predecessors=()):
    """A* alignment of *trace* against *trie_node*.

    When *partial* is True and a *target* is given, the search may stop as
    soon as the target node is reached — unconsumed trace events are left
    for a higher-level call.  When False (default), ALL trace events must
    be consumed before the goal is accepted.

    *required_predecessors* is a tuple of LocatedActivity objects that must
    have been synced at the target's location immediately before the target.

    Returns the cheapest Alignment.  If no alignment is possible (empty
    trie, impossible target) returns an Alignment consisting only of log
    moves for the remaining trace – never None.
    """
    if trie_node is None:
        result = Alignment()
        for event in trace:
            result = result.move_on_log_skip_model(event)
        return result

    start_node = trie_node
    counter = 0



    def _priority(path, cnt, trace_idx):
        return (path.get_cost(), -trace_idx if partial else 0, cnt)

    queue = [(*_priority(Alignment(), counter, 0), start_node, 0, Alignment())]
    visited = set()

    while queue:
        _, _, _, current_node, trace_idx, path = heapq.heappop(queue)


        if target:

            if (not current_node.is_root()
                    and hasattr(current_node.label, 'activity')
                    and current_node.label == target
                    and (partial or trace_idx == len(trace))):
                if required_predecessors:
                    if _check_predecessors(path, required_predecessors, target):
                        return path
                else:
                    return path
        else:
            if trace_idx == len(trace) and (current_node.is_leaf() or current_node.is_root()):
                return path

        state_id = (id(current_node), trace_idx)
        if state_id in visited:
            continue
        visited.add(state_id)


        if trace_idx < len(trace) and current_node.has_child(trace[trace_idx]):
            next_node = current_node.get_child(trace[trace_idx])
            counter += 1
            new_path = path.sync_move(trace[trace_idx])
            heapq.heappush(queue, (
                *_priority(new_path, counter, trace_idx + 1),
                next_node,
                trace_idx + 1,
                new_path,
            ))


        for next_node in current_node.get_children():
            counter += 1
            new_path = path.move_on_model_skip_log(next_node.label)
            heapq.heappush(queue, (
                *_priority(new_path, counter, trace_idx),
                next_node,
                trace_idx,
                new_path,
            ))


        if trace_idx < len(trace):
            counter += 1
            new_path = path.move_on_log_skip_model(trace[trace_idx])
            heapq.heappush(queue, (
                *_priority(new_path, counter, trace_idx + 1),
                current_node,
                trace_idx + 1,
                new_path,
            ))


    result = Alignment()
    for event in trace:
        result = result.move_on_log_skip_model(event)
    return result
