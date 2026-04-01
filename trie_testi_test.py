import unittest

from algo.network import Network
from algo.alignment_node import LocatedActivity, NetworkNode, Trie
from algo.alignments.trie_builder import TrieBuilder
from algo.utility.event_log_splitter import EventLogSplitter
from algo.datastructure.located_activity import EntryPoint

class TrieTestiTest(unittest.TestCase):
    def _get_training_traces(self, c: bool):
        return [
            [
                LocatedActivity("A", "c" if c else "n1"),
                LocatedActivity("B", "c" if c else "n1"),
                LocatedActivity("E", "c" if c else "n2"),
                LocatedActivity("F", "c" if c else "n2"),
                LocatedActivity("H", "c" if c else "n2"),
                LocatedActivity("G", "c" if c else "n4")
            ],
            [
                LocatedActivity("A", "c" if c else "n1"),
                LocatedActivity("C", "c" if c else "n1"),
                LocatedActivity("D", "c" if c else "n3"),
                LocatedActivity("G", "c" if c else "n4")
            ]
        ]

    def _get_validation_trace(self, c: bool):
        return [
            LocatedActivity("A", "c" if c else "n1"),
            LocatedActivity("B", "c" if c else "n1"),
            LocatedActivity("D", "c" if c else "n3"),
            LocatedActivity("F", "c" if c else "n2"),
            LocatedActivity("G", "c" if c else "n4")
        ]

    def _get_training_traces_context_sensitive(self, c: bool):
        return [
            [
                LocatedActivity("A", "c" if c else "n1"),
                LocatedActivity("C", "c" if c else "n1"),
                LocatedActivity("E", "c" if c else "n2")
            ],
            [
                LocatedActivity("B", "c" if c else "n1"),
                LocatedActivity("C", "c" if c else "n1"),
                LocatedActivity("D", "c" if c else "n2"),
            ]
        ]

    def _get_validation_trace_context_sensitive(self, c: bool):
        return [
            LocatedActivity("A", "c" if c else "n1"),
            LocatedActivity("C", "c" if c else "n1"),
            LocatedActivity("D", "c" if c else "n2")
        ]

    def _get_training_traces_alternating(self, c: bool):
        return [
            [
                LocatedActivity("A", "c" if c else "n1"),
                LocatedActivity("A2", "c" if c else "n1"),
                LocatedActivity("B", "c" if c else "n2"),
                LocatedActivity("C", "c" if c else "n2"),
                LocatedActivity("D", "c" if c else "n1"),
                LocatedActivity("D2", "c" if c else "n1"),
                LocatedActivity("E", "c" if c else "n2"),
                LocatedActivity("E2", "c" if c else "n2"),
                LocatedActivity("E3", "c" if c else "n1"),
                LocatedActivity("E4", "c" if c else "n1")
            ]
        ]

    def _get_validation_traces_alternating(self, c: bool):
        return [
            LocatedActivity("A", "c" if c else "n1"),
            LocatedActivity("B", "c" if c else "n2"),
            LocatedActivity("C", "c" if c else "n2"),
            LocatedActivity("D", "c" if c else "n1"),
            LocatedActivity("D2", "c" if c else "n1"),
            LocatedActivity("E", "c" if c else "n2"),

            LocatedActivity("E3", "c" if c else "n1"),
            LocatedActivity("E4", "c" if c else "n1")
        ]

    def _get_training_traces_skip_after_entry_point(self, c: bool):
        return [
            [
                LocatedActivity("A", "c" if c else "n1"),
                LocatedActivity("C", "c" if c else "n1"),
                LocatedActivity("D", "c" if c else "n2"),
                LocatedActivity("D2", "c" if c else "n2"),
                LocatedActivity("D3", "c" if c else "n2"),
                LocatedActivity("D4", "c" if c else "n2"),
                LocatedActivity("F", "c" if c else "n2")
            ],
            [
                LocatedActivity("A", "c" if c else "n1"),
                LocatedActivity("B", "c" if c else "n1"),
                LocatedActivity("E", "c" if c else "n2"),
                LocatedActivity("E2", "c" if c else "n2"),
                LocatedActivity("E3", "c" if c else "n2"),
                LocatedActivity("F", "c" if c else "n2")
            ]
        ]

    def _get_validation_traces_skip_after_entry_point(self, c: bool):
        return [
            LocatedActivity("A", "c" if c else "n1"),
            LocatedActivity("B", "c" if c else "n1"),
            LocatedActivity("D2", "c" if c else "n2"),
            LocatedActivity("D4", "c" if c else "n2"),
            LocatedActivity("F", "c" if c else "n2")
        ]

    def _get_training_traces_skip_node_simple(self, c: bool):
        return [
            [
                LocatedActivity("A", "c" if c else "n1"),
                LocatedActivity("B", "c" if c else "n2"),
                LocatedActivity("B2", "c" if c else "n2"),
                LocatedActivity("C", "c" if c else "n1"),
            ]
        ]

    def _get_real_dataset(self, c: bool):
        event_log_splitter = EventLogSplitter("gt/test/datasets/Sepsis.xes", location_key="org:group")
        training = event_log_splitter.get_training_data(1)
        result = []

        for key in training.traces:
            trace = []
            for event in training.traces[key]:
                trace.append(LocatedActivity(event.activity, "c" if c else event.location))
            result.append(trace)
        return result

    def _get_real_dataset_validation(self, c: bool):
        event_log_splitter = EventLogSplitter("gt/test/datasets/Sepsis.xes", location_key="org:group")
        training = event_log_splitter.get_test_data(1)
        result = []

        for key in training.traces:
            trace = []
            for event in training.traces[key]:
                trace.append(LocatedActivity(event.activity, "c" if c else event.location))
            result.append(trace)
        return result[0]


    def _get_validation_traces_skip_node_simple(self, c: bool):
        return [
            LocatedActivity("A", "c" if c else "n1"),
            LocatedActivity("C", "c" if c else "n1"),
        ]

    def _get_training_traces_skip_node(self, c: bool):
        return [
            [
                LocatedActivity("A", "c" if c else "n1"),
                LocatedActivity("B", "c" if c else "n2"),
                LocatedActivity("C", "c" if c else "n3"),
                LocatedActivity("D", "c" if c else "n1"),
                LocatedActivity("E", "c" if c else "n2"),
                LocatedActivity("F", "c" if c else "n1"),
            ]
        ]

    def _get_validation_traces_skip_node(self, c: bool):
        return [
            LocatedActivity("A", "c" if c else "n1"),
            LocatedActivity("C", "c" if c else "n3"),
        ]

    def _get_validation_traces_skip_node2(self, c: bool):
        return [
            LocatedActivity("A", "c" if c else "n1"),
            LocatedActivity("B", "c" if c else "n2"),
            LocatedActivity("C", "c" if c else "n3"),
            LocatedActivity("D", "c" if c else "n1"),
            LocatedActivity("F", "c" if c else "n1"),
        ]



    def _run(self, training_trace, validation_trace, context_depth=1):
        trie_builders = {}
        last_event = None
        for trace in training_trace:
            trace_history = []
            for located_activity in trace:
                if located_activity.location not in trie_builders:
                    trie_builders[located_activity.location] = TrieBuilder(Trie())
                if last_event and last_event.location != located_activity.location:
                    trie_builders[located_activity.location].reset()
                    if context_depth > 0:
                        upstream_events = [e for e in trace_history if e.location == last_event.location]
                        predecessors = tuple(upstream_events[-(context_depth + 1):-1]) if len(upstream_events) > 1 else ()
                        entry = EntryPoint(last_event.activity, last_event.location, predecessors)
                    else:
                        entry = last_event
                    trie_builders[located_activity.location].insert(entry)
                trie_builders[located_activity.location].insert(located_activity)
                trace_history.append(located_activity)
                last_event = located_activity
            for trie_id in trie_builders:
                trie_builders[trie_id].reset()
            last_event = None

        network = Network()
        for key in trie_builders:
            NetworkNode(trie_builders[key].root_trie, network, key)

        alignments = []
        for i, located_activity in enumerate(validation_trace):
            alignment = network.get_node(located_activity.location).process_event(located_activity, i)
            alignments.append(alignment)
            print(alignment)
        return alignments

    @staticmethod
    def _normalize_element(element):
        """Strip location from an AlignmentElement so centralized ('c') and
        distributed ('n1', 'n2', …) elements become comparable."""
        from algo.datastructure.alignment_constants import SKIP
        m = element.model.activity if element.model != SKIP else ">>"
        l = element.log.activity if element.log != SKIP else ">>"
        return (m, l)

    def _assert_equality_of_alignments(self, alignments_decentral, alignments_central):
        from collections import Counter
        fail = False
        for i in range(len(alignments_decentral)):
            self.assertEqual(alignments_central[i].get_cost(), alignments_decentral[i].get_cost(),
                             f"Event {i}: cost mismatch")
            central_norm = Counter(self._normalize_element(e) for e in alignments_central[i].elements)
            decentral_norm = Counter(self._normalize_element(e) for e in alignments_decentral[i].elements)
            if central_norm != decentral_norm:
                print(
                    f"Alignments do not match at event {i}.\nDecentral:\n{alignments_decentral[i]}\nCentral:\n{alignments_central[i]}")
                print(f"Central elements:  {central_norm}")
                print(f"Decentral elements: {decentral_norm}")
                fail = True
        self.assertFalse(fail)

    def test_example(self):
        alignments_decentral = self._run(self._get_training_traces(False), self._get_validation_trace(False))
        alignments_central = self._run(self._get_training_traces(True), self._get_validation_trace(True))
        self._assert_equality_of_alignments(alignments_decentral, alignments_central)

    def test_alternating(self):
        alignments_decentral = self._run(self._get_training_traces_alternating(False),
                                         self._get_validation_traces_alternating(False))
        alignments_central = self._run(self._get_training_traces_alternating(True),
                                       self._get_validation_traces_alternating(True))
        self._assert_equality_of_alignments(alignments_decentral, alignments_central)

    def test_validation_traces_skip_after_entry_point(self):
        alignments_decentral = self._run(self._get_training_traces_skip_after_entry_point(False),
                                         self._get_validation_traces_skip_after_entry_point(False))
        alignments_central = self._run(self._get_training_traces_skip_after_entry_point(True),
                                       self._get_validation_traces_skip_after_entry_point(True))
        self._assert_equality_of_alignments(alignments_decentral, alignments_central)

    def test_validation_traces_skip_node(self):
        alignments_decentral = self._run(self._get_training_traces_skip_node(False),
                                         self._get_validation_traces_skip_node(False))
        alignments_central = self._run(self._get_training_traces_skip_node(True),
                                       self._get_validation_traces_skip_node(True))
        self._assert_equality_of_alignments(alignments_decentral, alignments_central)

    def test_validation_traces_skip_node2(  self):
        alignments_decentral = self._run(self._get_training_traces_skip_node(False),
                                         self._get_validation_traces_skip_node2(False))
        alignments_central = self._run(self._get_training_traces_skip_node(True),
                                       self._get_validation_traces_skip_node2(True))
        self._assert_equality_of_alignments(alignments_decentral, alignments_central)

    def test_validation_traces_skip_node_simple(self):
        alignments_decentral = self._run(self._get_training_traces_skip_node_simple(False),
                                         self._get_validation_traces_skip_node_simple(False))
        alignments_central = self._run(self._get_training_traces_skip_node_simple(True),
                                       self._get_validation_traces_skip_node_simple(True))
        self._assert_equality_of_alignments(alignments_decentral, alignments_central)

    def test_context_sensitive(self):
       alignments_decentral = self._run(self._get_training_traces_context_sensitive(False),
                                        self._get_validation_trace_context_sensitive(False))
       alignments_central = self._run(self._get_training_traces_context_sensitive(True),
                                      self._get_validation_trace_context_sensitive(True))
       self._assert_equality_of_alignments(alignments_decentral, alignments_central)


if __name__ == "__main__":
    unittest.main()