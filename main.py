"""
main.py – Isolate the conceptual error in the decentral alignment approach
           using the real Sepsis.xes event log.
"""

import os
import sys
from typing import List

from algo.alignment_node import NetworkNode
from algo.alignments.trie_builder import TrieBuilder
from algo.datastructure.alignment import Alignment
from algo.datastructure.located_activity import LocatedActivity
from algo.datastructure.trie import Trie
from algo.network import Network
from algo.utility.event_log_splitter import EventLogSplitter


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SEPSIS_XES = os.path.join(PROJECT_ROOT, "algo", "Sepsis.xes")
LOCATION_KEY = "org:group"
N_TRAINING = 5
N_TEST = 1



def load_traces() -> tuple:
    splitter = EventLogSplitter(SEPSIS_XES, location_key=LOCATION_KEY)
    training = splitter.get_training_data(N_TRAINING)
    test = splitter.get_test_data(N_TEST)
    return training, test


def to_located_activities(event_log) -> List[List[LocatedActivity]]:
    traces = []
    for case_id in event_log.traces:
        trace = []
        for event in event_log.traces[case_id]:
            trace.append(LocatedActivity(event.activity, event.location))
        traces.append(trace)
    return traces


def build_network(training_traces: List[List[LocatedActivity]]) -> Network:
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
    return network


def replay(network: Network, validation_trace: List[LocatedActivity]) -> List[Alignment]:
    alignments = []
    for i, located_activity in enumerate(validation_trace):
        alignment = network.get_node(located_activity.location).process_event(located_activity, i)
        alignments.append(alignment)
    return alignments


def print_trie(trie: Trie, indent: int = 0, max_depth: int = 6):
    if indent > max_depth:
        print(" " * indent + "…")
        return
    print(" " * indent + str(trie.label))
    for child in trie.children:
        print_trie(child, indent + 2, max_depth)


if __name__ == "__main__":
    training_log, test_log = load_traces()

    training_traces = to_located_activities(training_log)
    test_traces = to_located_activities(test_log)
    
    all_locations = {la.location for trace in training_traces for la in trace}
    print(f"{len(training_traces)} training trace(s), locations: {sorted(all_locations)}")
    for i, trace in enumerate(training_traces):
        print(f"  Trace {i} ({len(trace)} events): {' → '.join(str(la) for la in trace)}")

    network = build_network(training_traces)
    for node_id in sorted(network.nodes):
        print(f"\nNode '{node_id}' trie:")
        print_trie(network.get_node(node_id).model, indent=2)


    validation_trace = test_traces[0]
    case_id = list(test_log.traces.keys())[0]

    print(f"\nTest case '{case_id}' – {len(validation_trace)} events:")
    for i, la in enumerate(validation_trace):
        print(f"  [{i:2d}] {la}")

    print("\nRunning decentral alignment …")
    alignments = replay(network, validation_trace)

    for i, alignment in enumerate(alignments):
        print(f"\nStep {i} ({validation_trace[i]}):")
        print(alignment)
