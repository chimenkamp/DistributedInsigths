class AlignmentElement:
    __slots__ = ('model', 'log')

    def __init__(self, model, log):
        self.model = model
        self.log = log

    def __str__(self):
        return f"Model: {self.model} | Log: {self.log}"

    def __repr__(self):
        return f"({self.model}, {self.log})"

    def __lt__(self, other):
        return False

    def __hash__(self):
        return hash((self.model, self.log))

    def __eq__(self, other):
        if not isinstance(other, AlignmentElement):
            return NotImplemented
        return self.model == other.model and self.log == other.log