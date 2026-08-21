from __future__ import annotations

import heapq
from typing import Dict, List, Tuple


Entry = Tuple[int, bytes, bytes]
Key = Tuple[bytes, bytes]


class _HeapEntry:
    """Heap item ordered by (value asc, key1 desc, key2 desc).

    heapq is a min-heap, so the item that compares smallest is popped first.
    Reversing only the key comparison means that value ties resolve to the
    LARGEST ``(key1, key2)`` -- the tie-break used by the reference BPE
    implementation (among equally frequent pairs, the lexicographically
    largest ``(token1, token2)`` is merged first).
    """

    __slots__ = ("value", "key1", "key2")

    def __init__(self, value: int, key1: bytes, key2: bytes) -> None:
        self.value = value
        self.key1 = key1
        self.key2 = key2

    def __lt__(self, other: _HeapEntry) -> bool:
        if self.value != other.value:
            return self.value < other.value
        if self.key1 != other.key1:
            return self.key1 > other.key1
        return self.key2 > other.key2


class MinValueStore:
    """Store ``(value, key1, key2)`` entries and retrieve the smallest value.

    ``(key1, key2)`` is unique.  Updating a key changes only its value.  The
    heap uses lazy deletion, so outdated heap entries are ignored by
    :meth:`pop_min`.

    Ties on ``value`` are resolved by the LARGEST ``key1`` and then ``key2``
    (matching the reference BPE tie-break: among equally frequent pairs, the
    lexicographically largest ``(token1, token2)`` is merged first).
    """

    def __init__(self) -> None:
        self._heap: List[_HeapEntry] = []
        self._values: Dict[Key, int] = {}

    def __contains__(self, key: Key) -> bool:
        return key in self._values

    def __getitem__(self, key: Key) -> int:
        return self._values[key]

    def add(self, value: int, key1: bytes, key2: bytes) -> None:
        """Add an entry.

        Raises:
            KeyError: if ``(key1, key2)`` already exists.
        """
        key = (key1, key2)
        if key in self._values:
            raise KeyError(f"key already exists: {key}")

        self._values[key] = value
        heapq.heappush(self._heap, _HeapEntry(value, key1, key2))

    def update(self, value: int, key1: bytes, key2: bytes) -> None:
        """Update the value for ``(key1, key2)``.

        Raises:
            KeyError: if the key does not exist.
        """
        key = (key1, key2)
        if key not in self._values:
            raise KeyError(f"key does not exist: {key}")

        self._values[key] = value
        heapq.heappush(self._heap, _HeapEntry(value, key1, key2))

    def remove(self, key1: bytes, key2: bytes) -> None:
        """Remove the entry for ``(key1, key2)``.

        The corresponding heap entry is left in place and skipped later by
        :meth:`pop_min` (lazy deletion).

        Raises:
            KeyError: if the key does not exist.
        """
        key = (key1, key2)
        if key not in self._values:
            raise KeyError(f"key does not exist: {key}")

        del self._values[key]

    def pop_min(self) -> Entry:
        """Remove and return the current smallest ``(value, key1, key2)``.

        Among entries with the same value, the largest ``(key1, key2)`` is
        returned first.

        Raises:
            IndexError: if the store is empty.
        """
        while self._heap:
            entry = heapq.heappop(self._heap)
            key = (entry.key1, entry.key2)
            if self._values.get(key) == entry.value:
                del self._values[key]
                return entry.value, entry.key1, entry.key2

        raise IndexError("pop_min from empty MinValueStore")

    def __len__(self) -> int:
        return len(self._values)

    def __bool__(self) -> bool:
        return bool(self._values)
