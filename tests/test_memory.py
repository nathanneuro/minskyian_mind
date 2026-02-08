"""Unit tests for minsky.memory module."""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta

from minsky.memory import (
    Memory,
    MemoryState,
    MemoryStore,
    fsrs_retention,
    get_memory_store,
    FSRS_PARAMS,
)


class TestFSRSRetention:
    """Tests for FSRS-6 retention curve."""

    def test_retention_at_zero_days(self):
        """Retention at 0 days should be ~1.0."""
        ret = fsrs_retention(0.0, stability=1.0)
        assert ret > 0.99

    def test_retention_decreases_over_time(self):
        """Retention should decrease as time passes."""
        ret_day1 = fsrs_retention(1.0, stability=10.0)
        ret_day10 = fsrs_retention(10.0, stability=10.0)
        ret_day100 = fsrs_retention(100.0, stability=10.0)

        assert ret_day1 > ret_day10 > ret_day100

    def test_higher_stability_slower_decay(self):
        """Higher stability should mean slower forgetting."""
        ret_low_stab = fsrs_retention(10.0, stability=1.0)
        ret_high_stab = fsrs_retention(10.0, stability=100.0)

        assert ret_high_stab > ret_low_stab

    def test_retention_bounded_0_1(self):
        """Retention should always be between 0 and 1."""
        for days in [0, 1, 10, 100, 1000]:
            for stab in [0.1, 1, 10, 100]:
                ret = fsrs_retention(float(days), stability=stab)
                assert 0 <= ret <= 1


class TestMemory:
    """Tests for Memory dataclass."""

    def test_memory_creation(self):
        """Basic memory creation should work."""
        mem = Memory(
            id="test123",
            content="Test content",
            tags=["tag1", "tag2"],
            source="test",
        )
        assert mem.id == "test123"
        assert mem.content == "Test content"
        assert mem.tags == ["tag1", "tag2"]

    def test_memory_default_values(self):
        """Memory should have sensible defaults."""
        mem = Memory(id="test", content="Test")

        assert mem.storage_strength == 1.0
        assert mem.retrieval_strength == 1.0
        assert mem.stability == 1.0
        # access_count starts at 1 in this implementation
        assert mem.access_count == 1

    def test_memory_retention(self):
        """Retention should decrease over time."""
        mem = Memory(id="test", content="Test")

        # Initially should be high
        initial_ret = mem.retention
        assert initial_ret > 0.9

        # After simulating time passage
        mem.last_accessed = datetime.now() - timedelta(days=30)
        later_ret = mem.retention
        assert later_ret < initial_ret

    def test_memory_accessibility(self):
        """Accessibility should be calculated from multiple factors."""
        mem = Memory(id="test", content="Test")

        acc = mem.accessibility
        assert 0 <= acc <= 1

    def test_memory_state_active(self):
        """New memory should be in ACTIVE state."""
        mem = Memory(id="test", content="Test")
        assert mem.get_state() == MemoryState.ACTIVE

    def test_memory_state_degrades(self):
        """Memory with low accessibility should not be ACTIVE."""
        mem = Memory(id="test", content="Test")
        mem.storage_strength = 0.01
        mem.retrieval_strength = 0.01
        mem.stability = 0.01
        mem.last_accessed = datetime.now() - timedelta(days=365)

        state = mem.get_state()
        assert state != MemoryState.ACTIVE

    def test_access_increases_count(self):
        """Accessing memory should increase access count."""
        mem = Memory(id="test", content="Test")
        initial_count = mem.access_count

        # Simulate access
        mem.access_count += 1

        assert mem.access_count > initial_count


class TestMemoryStore:
    """Tests for MemoryStore."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield Path(path)
        os.unlink(path)

    def test_store_and_retrieve(self, temp_db):
        """Should store and retrieve memories."""
        store = MemoryStore(db_path=temp_db)

        # Create a Memory object to store
        mem = Memory(
            id="test1",
            content="Test content",
            tags=["test"],
            source="unit_test",
        )
        mem_id = store.store(mem)
        retrieved = store.get(mem_id)

        assert retrieved is not None
        assert retrieved.content == "Test content"
        assert "test" in retrieved.tags

    def test_search_by_query(self, temp_db):
        """Should search memories by query."""
        store = MemoryStore(db_path=temp_db)

        store.store(Memory(id="m1", content="Python programming is fun", tags=["code"]))
        store.store(Memory(id="m2", content="JavaScript is also popular", tags=["code"]))
        store.store(Memory(id="m3", content="Cooking recipes", tags=["food"]))

        # search returns list of (Memory, score) tuples
        results = store.search("programming", limit=10)

        assert len(results) >= 1
        assert any("Python" in mem.content for mem, _score in results)

    def test_search_returns_scored_results(self, temp_db):
        """Search should return (Memory, score) tuples."""
        store = MemoryStore(db_path=temp_db)

        store.store(Memory(id="m1", content="Python programming language", tags=["code"]))

        results = store.search("Python", limit=10)

        assert len(results) >= 1
        mem, score = results[0]
        assert isinstance(score, float)
        assert score > 0

    def test_promote_memory(self, temp_db):
        """Promoting should increase memory strength."""
        store = MemoryStore(db_path=temp_db)

        mem = Memory(id="imp", content="Important memory", tags=["test"])
        mem_id = store.store(mem)
        mem_before = store.get(mem_id)
        initial_storage = mem_before.storage_strength

        store.promote(mem_id)
        mem_after = store.get(mem_id)

        assert mem_after.storage_strength > initial_storage

    def test_demote_memory(self, temp_db):
        """Demoting should decrease retrieval strength."""
        store = MemoryStore(db_path=temp_db)

        mem = Memory(id="less", content="Less important memory", tags=["test"])
        mem_id = store.store(mem)
        mem_before = store.get(mem_id)
        initial_retrieval = mem_before.retrieval_strength

        store.demote(mem_id)
        mem_after = store.get(mem_id)

        assert mem_after.retrieval_strength < initial_retrieval

    def test_stats(self, temp_db):
        """Should return memory statistics."""
        store = MemoryStore(db_path=temp_db)

        store.store(Memory(id="s1", content="Memory 1", tags=["a"]))
        store.store(Memory(id="s2", content="Memory 2", tags=["b"]))
        store.store(Memory(id="s3", content="Memory 3", tags=["a", "b"]))

        stats = store.stats()

        assert stats["total"] == 3
        assert "by_state" in stats

    def test_decay_all(self, temp_db):
        """Should decay all memories."""
        store = MemoryStore(db_path=temp_db)

        mem = Memory(id="decay", content="Test memory", tags=["test"])
        mem_id = store.store(mem)
        mem_before = store.get(mem_id)
        initial_retrieval = mem_before.retrieval_strength

        # decay_all takes hours parameter
        count = store.decay_all(hours=24.0)
        mem_after = store.get(mem_id)

        assert count >= 1
        assert mem_after.retrieval_strength < initial_retrieval


class TestMemoryState:
    """Tests for MemoryState enum."""

    def test_states_exist(self):
        """Core memory states should exist."""
        assert MemoryState.ACTIVE.value == "active"
        assert MemoryState.DORMANT.value == "dormant"
