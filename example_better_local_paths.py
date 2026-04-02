from algo.alignment_node import NetworkNode
from algo.alignments.trie_builder import TrieBuilder
from algo.datastructure.alignment import Alignment
from algo.datastructure.trie import Trie
from algo.network import Network
from algo.datastructure.located_activity import LocatedActivity, EntryPoint


def build_network(training_traces, context_depth=1):
    """Build a distributed network of per-node tries from training traces."""
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


def replay(network, validation_trace):
    """Replay a validation trace and return per-event alignments."""
    alignments = []
    for i, ev in enumerate(validation_trace):
        alignment = network.get_node(ev.location).process_event(ev, i)
        alignments.append(alignment)
    return alignments


def print_alignment(label, alignment):
    print(f"  {label}: cost={alignment.get_cost()}")
    for e in alignment.elements:
        print(f"    {e}")

print("=" * 72)
print("EXAMPLE: Distributed finds lower cost than centralized")
print("=" * 72)


training = [
    [
        LocatedActivity("Register", "HQ"),
        LocatedActivity("BloodTest", "LAB"),
        LocatedActivity("XRay", "LAB"),
        LocatedActivity("Diagnose", "HQ"),
    ],
    [
        LocatedActivity("Register", "HQ"),
        LocatedActivity("XRay", "LAB"),
        LocatedActivity("BloodTest", "LAB"),
        LocatedActivity("Diagnose", "HQ"),
    ],
]

validation_distributed = [
    LocatedActivity("Register", "HQ"),
    LocatedActivity("BloodTest", "LAB"),
    LocatedActivity("Diagnose", "HQ"),
]
validation_centralized = [
    LocatedActivity("Register", "c"),
    LocatedActivity("BloodTest", "c"),
    LocatedActivity("Diagnose", "c"),
]


training_centralized = [
    [LocatedActivity(ev.activity, "c") for ev in trace]
    for trace in training
]

print("\nTraining trace 1:", " → ".join(str(e) for e in training[0]))
print("Training trace 2:", " → ".join(str(e) for e in training[1]))
print("Validation:       ", " → ".join(str(e) for e in validation_distributed))


de_network = build_network(training)
ce_network = build_network(training_centralized)


de_alignments = replay(de_network, validation_distributed)
ce_alignments = replay(ce_network, validation_centralized)

print("\n── Event-by-event comparison ──")
for i in range(len(de_alignments)):
    dc = de_alignments[i].get_cost()
    cc = ce_alignments[i].get_cost()
    marker = " ← MISMATCH" if dc != cc else ""
    print(f"\nEvent {i}: {validation_distributed[i]}")
    print_alignment("Centralized", ce_alignments[i])
    print_alignment("Distributed", de_alignments[i])
    if dc != cc:
        diff = cc - dc
        print(f"  ** Distributed is {'LOWER' if diff > 0 else 'HIGHER'} by {abs(diff)} **")

print("\n\n" + "=" * 72)
print("EXAMPLE 2: Skipped intermediate at remote location")
print("=" * 72)

training2 = [
    [
        LocatedActivity("A", "n1"),
        LocatedActivity("B", "n2"),
        LocatedActivity("C", "n2"),
        LocatedActivity("D", "n1"),
    ],
    [
        LocatedActivity("A", "n1"),
        LocatedActivity("C", "n2"),
        LocatedActivity("B", "n2"),
        LocatedActivity("D", "n1"),
    ],
]

validation2_distributed = [
    LocatedActivity("A", "n1"),
    LocatedActivity("C", "n2"),
    LocatedActivity("D", "n1"),
]
validation2_centralized = [
    LocatedActivity("A", "c"),
    LocatedActivity("C", "c"),
    LocatedActivity("D", "c"),
]

training2_centralized = [
    [LocatedActivity(ev.activity, "c") for ev in trace]
    for trace in training2
]

print("\nTraining trace 1:", " → ".join(str(e) for e in training2[0]))
print("Training trace 2:", " → ".join(str(e) for e in training2[1]))
print("Validation:       ", " → ".join(str(e) for e in validation2_distributed))

de_net2 = build_network(training2)
ce_net2 = build_network(training2_centralized)

de_al2 = replay(de_net2, validation2_distributed)
ce_al2 = replay(ce_net2, validation2_centralized)

print("\n── Event-by-event comparison ──")
for i in range(len(de_al2)):
    dc = de_al2[i].get_cost()
    cc = ce_al2[i].get_cost()
    print(f"\nEvent {i}: {validation2_distributed[i]}")
    print_alignment("Centralized", ce_al2[i])
    print_alignment("Distributed", de_al2[i])
    if dc != cc:
        diff = cc - dc
        print(f"  ** Distributed is {'LOWER' if diff > 0 else 'HIGHER'} by {abs(diff)} **")