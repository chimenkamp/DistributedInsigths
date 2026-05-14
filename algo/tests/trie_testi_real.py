import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

try:
    from algo.utility.event_log_splitter import EventLogSplitter
except ModuleNotFoundError as exc:
    EventLogSplitter = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from algo.alignment_node import LocatedActivity, NetworkNode, Trie
from algo.alignments.trie_builder import TrieBuilder
from algo.datastructure.alignment_constants import SKIP
from algo.network import Network


SEPSIS_XES = Path(__file__).parents[1] / "Sepsis.xes"
TRAINING_RECORDS = 100
VALIDATION_RECORDS = 1
MAX_VALIDATION_EVENTS = 4


class SepsisAlignmentTest(unittest.TestCase):
    @unittest.skipUnless(SEPSIS_XES.exists(), f"Missing dataset: {SEPSIS_XES}")
    @unittest.skipIf(EventLogSplitter is None, f"Missing dependency: {IMPORT_ERROR}")
    def test_sepsis_distributed_has_same_cost_profile_as_central(self):
        training_data, validation_data = self._get_sepsis_logs()
        training_distributed = self._to_located_traces(training_data, central=False)
        validation_distributed = self._to_located_traces(validation_data, central=False)[0][:MAX_VALIDATION_EVENTS]

        training_central = self._to_located_traces(training_data, central=True)
        validation_central = self._to_located_traces(validation_data, central=True)[0][:MAX_VALIDATION_EVENTS]

        alignments_distributed = self._run(training_distributed, validation_distributed)
        alignments_central = self._run(training_central, validation_central)

        self.assertEqual(len(alignments_central), len(alignments_distributed))
        for i, (central, distributed) in enumerate(zip(alignments_central, alignments_distributed)):
            if self._cost_profile(central) != self._cost_profile(distributed):
                self.fail(
                    self._failure_message(
                        i,
                        validation_distributed,
                        validation_central,
                        distributed,
                        central,
                    )
                )

    def _get_sepsis_logs(self):
        splitter = EventLogSplitter(str(SEPSIS_XES), location_key="org:group")
        return (
            splitter.get_training_data(TRAINING_RECORDS),
            splitter.get_test_data(VALIDATION_RECORDS),
        )

    def _to_located_traces(self, event_log, central: bool):
        traces = []
        for case_id in event_log.traces:
            trace = []
            for event in event_log.traces[case_id]:
                location = "c" if central else event.activity
                trace.append(LocatedActivity(event.activity, location))
            traces.append(trace)
        return traces

    def _run(self, training_traces, validation_trace):
        trie_builders = {}
        last_event = None

        for trace in training_traces:
            for located_activity in trace:
                if located_activity.location not in trie_builders:
                    trie_builders[located_activity.location] = TrieBuilder(Trie())
                if last_event and last_event.location != located_activity.location:
                    trie_builders[located_activity.location].reset()
                    trie_builders[located_activity.location].insert(last_event)
                trie_builders[located_activity.location].insert(located_activity)
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
        return alignments

    def _cost_profile(self, alignment):
        return {
            "cost": alignment.get_cost(),
            "sync_moves": self._count_sync_moves(alignment),
            "model_moves": self._count_model_moves(alignment),
            "log_moves": self._count_log_moves(alignment),
            "total_moves": len(alignment.elements),
        }

    def _failure_message(
            self,
            event_index,
            validation_distributed,
            validation_central,
            distributed,
            central,
    ):
        distributed_prefix = validation_distributed[:event_index + 1]
        central_prefix = validation_central[:event_index + 1]
        return (
            f"Different alignment profile after event {event_index}: "
            f"{validation_distributed[event_index]}\n\n"
            f"Distributed prefix:\n{self._format_trace(distributed_prefix)}\n\n"
            f"Central prefix:\n{self._format_trace(central_prefix)}\n\n"
            f"Distributed profile: {self._cost_profile(distributed)}\n"
            f"{distributed}\n"
            f"Central profile: {self._cost_profile(central)}\n"
            f"{central}"
        )

    def _format_trace(self, trace):
        return "\n".join(f"{i}: {event}" for i, event in enumerate(trace))

    def _count_sync_moves(self, alignment):
        return len([
            element
            for element in alignment.elements
            if element.model != SKIP and element.log != SKIP
        ])

    def _count_model_moves(self, alignment):
        return len([
            element
            for element in alignment.elements
            if element.model != SKIP and element.log == SKIP
        ])

    def _count_log_moves(self, alignment):
        return len([
            element
            for element in alignment.elements
            if element.model == SKIP and element.log != SKIP
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
