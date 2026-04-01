class LocatedActivity:
    def __init__(self, activity, location):
        self.activity = activity
        self.location = location

    def __str__(self):
        return f"{self.activity}@{self.location}"

    def __repr__(self):
        return f"{self.activity}@{self.location}"

    def __eq__(self, other):
        if not isinstance(other, LocatedActivity):
            return False
        return self.activity == other.activity and self.location == other.location

    def same_activity(self, other):
        """Compare only the activity name, ignoring location."""
        if not isinstance(other, LocatedActivity):
            return False
        return self.activity == other.activity

    def __hash__(self):
        return hash((self.activity, self.location))


class EntryPoint(LocatedActivity):
    """A LocatedActivity enriched with predecessor context.

    *predecessors* is a tuple of up to *k* LocatedActivity objects that
    occurred at the upstream location immediately before this activity.
    Two entry points with the same activity/location but different
    predecessors are distinct, giving each per-node trie branch the
    upstream context needed to preserve path-sensitivity.
    """

    def __init__(self, activity, location, predecessors=None):
        super().__init__(activity, location)
        self.predecessors = tuple(predecessors) if predecessors else ()

    def __str__(self):
        if self.predecessors:
            prefix = "->".join(str(p) for p in self.predecessors)
            return f"[{prefix}]->{self.activity}@{self.location}"
        return f"[]->{self.activity}@{self.location}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, EntryPoint):
            if isinstance(other, LocatedActivity):
                return False
            return NotImplemented
        return (self.activity == other.activity
                and self.location == other.location
                and self.predecessors == other.predecessors)

    def __hash__(self):
        return hash((self.activity, self.location, self.predecessors))
