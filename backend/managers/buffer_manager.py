class BufferManager:
    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.history = []

    def add_turn(self, role, content):
        self.history.append({"role": role, "content": content})
        # Keep only the last N messages to prevent context overflow
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_messages(self):
        return self.history

    def clear(self):
        self.history = []