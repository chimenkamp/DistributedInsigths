import sys
from typing import List, Any, Set

from algo.datastructure.alignment import Alignment
from algo.alignments.alignment_calculator import calculate_alignment
from algo.datastructure.alignment_repsonse import AlignmentResponse
from algo.datastructure.located_activity import LocatedActivity, EntryPoint
from algo.network import Network
from algo.datastructure.trie import Trie


class NetworkNode:
    def __init__(self, model, network, node_id):
        self.node_id = node_id
        self.network: Network = network
        self.network.add_node(self.node_id, self)
        self.i = -1
        self.model: Trie = model
        self.observed_events = {}

        self.cache = {}

    def _get_trace(self, min_i=-1, max_i=sys.maxsize):
        filtered = {k: v for k, v in self.observed_events.items() if min_i < k < max_i}
        return [value for key, value in sorted(filtered.items())]

    def get_all_moves(self, i):
        return [v for k, v in sorted(self.observed_events.items()) if k < i]

    def get_alignment(self, target: LocatedActivity, i,
                      _visiting: Set[str] | None = None,
                      required_predecessors: tuple = ()) -> AlignmentResponse:
        """Called by a remote node requesting an upstream alignment."""
        response: AlignmentResponse = self.find_best_alignment(
            target, i, _visiting=_visiting,
            required_predecessors=required_predecessors,
        )
        if not response:
            return None

        return AlignmentResponse(
            max(self.i, response.timestamp),
            response.alignment,
            target,
            response.last_node,
        )

    def process_event(self, located_activity: LocatedActivity, i: int) -> Alignment:
        self.i = i
        self.observed_events[i] = located_activity

        self.cache = {}
        response = self.find_best_alignment(located_activity, i, is_start=True)



        responses = self._add_external_log_moves([response], i)
        return responses[0].alignment

    def find_best_alignment(
        self,
        target: LocatedActivity = None,
        i=sys.maxsize,
        is_start=False,
        _visiting: Set[tuple] | None = None,
        required_predecessors: tuple = (),
    ) -> Any:
        if _visiting is None:
            _visiting = set()

        _visiting = _visiting | {(self.node_id, target)}

        all_candidate_alignments: List[AlignmentResponse] = []
        rejected_entry_points: List = []
        for entry_point in self.model.get_children_containing_label(target):
            response, model = self._request_external_alignment(
                entry_point, i, _visiting,
            )
            if response is None:
                if isinstance(entry_point, EntryPoint) and entry_point.predecessors:
                    rejected_entry_points.append(entry_point)
                continue

            last_node = response.last_node
            trace = self._get_relevant_local_trace(i, is_start, last_node, response.timestamp, response.alignment)
            local_alignment = calculate_alignment(trace, model, target, partial=(not is_start),
                                                   required_predecessors=required_predecessors)

            if local_alignment is None:
                continue

            if local_alignment.contains_log_moves():
                last_node = self.node_id

            candidate_alignment = response.alignment + local_alignment
            all_candidate_alignments.append(
                AlignmentResponse(response.timestamp, candidate_alignment, target, last_node))



        if not all_candidate_alignments and rejected_entry_points:
            for entry_point in rejected_entry_points:
                response, model = self._request_external_alignment(
                    entry_point, i, _visiting, skip_predecessor_check=True,
                )
                if response is None:
                    continue

                last_node = response.last_node
                trace = self._get_relevant_local_trace(i, is_start, last_node, response.timestamp, response.alignment)
                local_alignment = calculate_alignment(trace, model, target, partial=(not is_start))

                if local_alignment is None:
                    continue

                if local_alignment.contains_log_moves():
                    last_node = self.node_id

                candidate_alignment = response.alignment + local_alignment
                all_candidate_alignments.append(
                    AlignmentResponse(response.timestamp, candidate_alignment, target, last_node))

        if not all_candidate_alignments:
            trace = self._get_relevant_local_trace(i, is_start, None, -1)
            result = calculate_alignment(trace, self.model, target, partial=(not is_start),
                                          required_predecessors=required_predecessors)
            if result is None:
                result = Alignment()
            return AlignmentResponse(-1, result, target, self.node_id)

        best_alignment_response = min(all_candidate_alignments, key=lambda x: x.alignment)

        return AlignmentResponse(
            best_alignment_response.timestamp,
            best_alignment_response.alignment,
            target,
            best_alignment_response.last_node,
        )

    def _add_external_log_moves(self, responses: List[AlignmentResponse], i):
        all_log_moves = []
        for node in self.network.get_all_nodes(self.node_id):
            all_log_moves.extend(node.get_all_moves(i))
        result = []
        for response in responses:
            response.alignment = response.alignment.append_missing_log_moves(all_log_moves)
            result.append(response)
        return result

    def _request_external_alignment(
        self,
        entry_point,
        i: int,
        _visiting: Set[str],
        skip_predecessor_check: bool = False,
    ) -> tuple[AlignmentResponse | None, Trie]:
        model = self.model

        if not hasattr(entry_point, 'location'):
            return AlignmentResponse(-1, Alignment(), entry_point, None), model


        if isinstance(entry_point, EntryPoint):
            upstream_target = LocatedActivity(entry_point.activity, entry_point.location)
        else:
            upstream_target = entry_point

        if self.node_id == entry_point.location:
            alignment_response = AlignmentResponse(-1, Alignment(), entry_point, None)
        elif (entry_point.location, upstream_target) in _visiting:
            return None, model
        else:
            cache_key = (entry_point, i)
            if cache_key in self.cache:
                alignment_response = self.cache[cache_key]
            else:
                preds = entry_point.predecessors if isinstance(entry_point, EntryPoint) else ()
                alignment_response = (
                    self.network
                    .get_node(entry_point.location)
                    .get_alignment(upstream_target, i, _visiting,
                                   required_predecessors=preds)
                )
                if alignment_response is not None:
                    self.cache[cache_key] = alignment_response

            model = self.model.get_child(entry_point)




            if (not skip_predecessor_check
                    and alignment_response is not None
                    and isinstance(entry_point, EntryPoint)
                    and entry_point.predecessors):
                if not self._predecessors_match(alignment_response.alignment, entry_point):
                    return None, model

        return alignment_response, model

    @staticmethod
    def _predecessors_match(upstream_alignment: Alignment, entry_point: EntryPoint) -> bool:
        """Check upstream alignment's sync moves end with expected predecessors.
        Returns True if no upstream context contradicts the expected predecessors."""
        from algo.datastructure.alignment_constants import SKIP
        sync_at_location = []
        for elem in upstream_alignment.elements:
            if elem.model != SKIP and elem.log != SKIP:
                if (hasattr(elem.model, 'location')
                        and elem.model.location == entry_point.location):
                    sync_at_location.append(elem.model)

        if sync_at_location and sync_at_location[-1].same_activity(
                LocatedActivity(entry_point.activity, entry_point.location)):
            sync_at_location = sync_at_location[:-1]
        k = len(entry_point.predecessors)
        available = len(sync_at_location)
        if available == 0:
            return True  # no upstream context to contradict
        compare_len = min(k, available)
        actual_tail = sync_at_location[-compare_len:]
        expected_tail = entry_point.predecessors[-compare_len:]
        for actual, expected in zip(actual_tail, expected_tail):
            if not actual.same_activity(expected):
                return False
        return True

    def _get_relevant_local_trace(self, i: int, is_start: bool, last_node, timestamp: int, upstream_alignment=None) -> list[Any]:
        """Get events that need to be aligned at this node.

        For the top-level call (is_start=True), includes ALL events across
        ALL nodes so the local A* has full global visibility — similar to
        the centralized approach.  For recursive GA calls, only local
        events are included.

        Events already consumed by the upstream alignment are removed.
        """
        max_time = i if not is_start else sys.maxsize
        if is_start:

            all_events = {}
            for node in self.network.nodes.values():
                for k, v in node.observed_events.items():
                    if k < max_time:
                        all_events[k] = v

            if i in self.observed_events:
                all_events[i] = self.observed_events[i]
            trace = [v for k, v in sorted(all_events.items())]
        else:
            trace = self._get_trace(-1, max_time)
        if upstream_alignment:
            for event in upstream_alignment.get_all_log_moves():
                try:
                    trace.remove(event)
                except ValueError:
                    pass
        return trace

    def get_activity(self) -> str:
        return self.node_id

    def get_edges(self) -> List[str]:
        return [child.get_locations() for child in self.model.get_children()]