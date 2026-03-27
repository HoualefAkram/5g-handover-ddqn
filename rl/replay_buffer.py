import pickle
from collections import deque
from pathlib import Path


class ReplayBuffer:
    def __init__(
        self,
        file_path: str = "cache/training/replay_buffer.pkl",
        max_len: int = 10000,
    ):
        self.file_path = Path(file_path)
        self.queue = deque(maxlen=max_len)
        self.load()

    def append(self, value):
        self.queue.append(value)

    def append_left(self, value):
        self.queue.appendleft(value)

    def clear(self):
        self.queue.clear()

    def save(self):
        """Save the buffer to disk."""
        # Create parent directories if they don't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "wb") as f:
            pickle.dump(self.queue, f)

    def load(self):
        """Load the buffer from disk if it exists."""
        if self.file_path.exists() and self.file_path.is_file():
            with open(self.file_path, "rb") as f:
                self.queue = pickle.load(f)
            print(f"Loaded {len(self.queue)} experiences from {self.file_path}")

    def delete_save(self):
        if self.file_path.exists() and self.file_path.is_file():
            self.file_path.unlink()
            print("Deleted saved replay buffer.")

    def __len__(self):
        return len(self.queue)
