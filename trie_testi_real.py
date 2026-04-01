import os
import sys
from typing import List, Optional

from algo.alignment_node import NetworkNode
from algo.alignments.trie_builder import TrieBuilder
from algo.datastructure.alignment import Alignment
from algo.datastructure.alignment import Alignment
from algo.datastructure.trie import Trie
from algo.network import Network


from algo.datastructure.located_activity import LocatedActivity, EntryPoint
from algo.utility.event_log_splitter import EventLogSplitter

LOCATION_KEY = "org:group"
N_TRAINING = 50
N_TEST = 1
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SEPSIS_XES = os.path.join("/Users/christianimenkamp/Documents/Data-Repository/Community/bpi-c-2013/BPI_Challenge_2013_open_problems.xes")

def load_traces() -> tuple:
    splitter = EventLogSplitter(SEPSIS_XES, location_key=LOCATION_KEY)
    training = splitter.get_training_data(N_TRAINING)
    test = splitter.get_test_data(N_TEST)
    return training, test

def to_located_activities(event_log, location_key: Optional[str] = None) -> List[List[LocatedActivity]]:
    traces = []
    
    for case_id in event_log.traces:
        trace = []
        for event in event_log.traces[case_id]:
            trace.append(LocatedActivity(event.activity, location=location_key if location_key else event.location))
        traces.append(trace)
    return traces

def build_network(training_traces: List[List[LocatedActivity]], context_depth: int = 1) -> Network:
    trie_builders = {}
    last_event = None

    for trace in training_traces:
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
    return network

def replay(network: Network, validation_trace: List[LocatedActivity]) -> List[Alignment]:
    alignments = []
    for i, located_activity in enumerate(validation_trace):
        alignment = network.get_node(located_activity.location).process_event(located_activity, i)
        alignments.append(alignment)
    return alignments



def _normalize_element(element):
    """Strip location from an AlignmentElement so centralized and distributed
    elements become comparable."""
    from algo.datastructure.alignment_constants import SKIP
    m = element.model.activity if element.model != SKIP else ">>"
    l = element.log.activity if element.log != SKIP else ">>"
    return (m, l)

def _assert_equality_of_alignments(alignments_decentral, alignments_central):
    from collections import Counter
    fail = False
    for i in range(len(alignments_decentral)):
        if alignments_central[i].get_cost() != alignments_decentral[i].get_cost():
            print(f"Event {i}: cost mismatch: central={alignments_central[i].get_cost()}, "
                  f"decentral={alignments_decentral[i].get_cost()}")
            fail = True
        central_norm = Counter(_normalize_element(e) for e in alignments_central[i].elements)
        decentral_norm = Counter(_normalize_element(e) for e in alignments_decentral[i].elements)
        if central_norm != decentral_norm:
            print(
                f"Alignments do not match at event {i}.\nDecentral:\n{alignments_decentral[i]}\nCentral:\n{alignments_central[i]}")
            print(f"Central elements:  {central_norm}")
            print(f"Decentral elements: {decentral_norm}")
            fail = True
    assert not fail

if __name__ == '__main__':
    training, test = load_traces()

    de_training = to_located_activities(training)
    de_test = to_located_activities(test)

    cent_training = to_located_activities(training, location_key="n1")
    cent_test = to_located_activities(test, location_key="n1")

    de_network = build_network(de_training)
    cent_network = build_network(cent_training)

    de_validation_trace = de_test[0]
    cent_validation_trace = cent_test[0]

    de_alignments = replay(de_network, de_validation_trace)
    cent_alignments = replay(cent_network, cent_validation_trace)

    _assert_equality_of_alignments(de_alignments, cent_alignments)

    print("Event\tDecentralized Alignment\tCentralized Alignment")
    for i in range(len(de_alignments)):
        print(f"{i}\t{de_alignments[i]}\t{cent_alignments[i]}")
    

