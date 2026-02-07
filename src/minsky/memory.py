"""Memory system inspired by Vestige.

Implements:
- FSRS-6 power law forgetting (not exponential)
- Dual-strength model (storage + retrieval)
- Hybrid search (keyword + semantic)
- Memory states based on accessibility

Motor commits to memory and queries it (as directed by Planning).
Sensory perceives the results of memory queries.
"""

import json
import math
import sqlite3
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MEMORY_DB = DATA_DIR / "memory.db"


# =============================================================================
# FSRS-6 Parameters (from Vestige)
# =============================================================================

# Default FSRS-6 parameters (optimized on 700M+ Anki reviews)
FSRS_PARAMS = {
    "w0": 0.40255,   # Initial stability
    "w1": 1.18385,
    "w2": 3.173,
    "w3": 15.69105,
    "w4": 7.1949,    # Difficulty weight
    "w5": 0.5345,
    "w6": 1.4604,
    "w7": 0.0046,
    "w8": 1.54575,
    "w9": 0.1192,
    "w10": 1.01925,
    "w11": 1.9395,
    "w12": 0.11,
    "w13": 0.29605,
    "w14": 2.2698,
    "w15": 0.2315,
    "w16": 2.9898,
    "w17": 0.51655,
    "w18": 0.6621,
    "w19": 0.5,      # Decay parameter base
    "w20": 0.8,      # Power law exponent
}


class MemoryState(Enum):
    """Memory accessibility states based on Vestige."""
    ACTIVE = "active"       # ≥70% accessibility
    DORMANT = "dormant"     # 40-70%
    SILENT = "silent"       # 10-40%
    UNAVAILABLE = "unavailable"  # <10%


@dataclass
class Memory:
    """A single memory entry with dual-strength model."""
    id: str
    content: str
    embedding: list[float] | None = None

    # Dual-strength (Bjork & Bjork, 1992)
    storage_strength: float = 1.0   # Only increases, never decreases
    retrieval_strength: float = 1.0 # Decays over time, restored by access

    # FSRS-6 scheduling
    stability: float = 1.0      # Time for retrievability to drop to 90%
    difficulty: float = 0.3     # 0-1, affects stability growth

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 1

    # Metadata
    tags: list[str] = field(default_factory=list)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_state(self) -> MemoryState:
        """Get current memory state based on accessibility."""
        acc = self.accessibility
        if acc >= 0.7:
            return MemoryState.ACTIVE
        elif acc >= 0.4:
            return MemoryState.DORMANT
        elif acc >= 0.1:
            return MemoryState.SILENT
        else:
            return MemoryState.UNAVAILABLE

    @property
    def retention(self) -> float:
        """Calculate current retention using FSRS-6 power law."""
        elapsed_days = (datetime.now() - self.last_accessed).total_seconds() / 86400
        return fsrs_retention(elapsed_days, self.stability)

    @property
    def accessibility(self) -> float:
        """Weighted combination of retention and strengths."""
        # accessibility = 0.5 × retention + 0.3 × retrieval + 0.2 × storage
        return (
            0.5 * self.retention +
            0.3 * min(1.0, self.retrieval_strength) +
            0.2 * min(1.0, self.storage_strength / 10.0)  # Normalize storage
        )

    def access(self) -> None:
        """Record an access (strengthens memory)."""
        # Testing effect: retrieval strengthens memory
        self.storage_strength += 0.1
        self.retrieval_strength = min(1.0, self.retrieval_strength + 0.3)

        # Update FSRS stability
        self.stability = update_stability(
            self.stability,
            self.difficulty,
            self.retention,
        )

        self.last_accessed = datetime.now()
        self.access_count += 1

    def decay(self, hours: float = 1.0) -> None:
        """Simulate time-based decay of retrieval strength."""
        decay_factor = 0.99 ** hours  # ~1% per hour
        self.retrieval_strength *= decay_factor


def fsrs_retention(elapsed_days: float, stability: float) -> float:
    """FSRS-6 power law forgetting curve.

    R(t, S) = (1 + factor × t / S)^(-w₂₀)
    where factor = 0.9^(-1/w₂₀) - 1
    """
    w20 = FSRS_PARAMS["w20"]
    factor = (0.9 ** (-1 / w20)) - 1

    if stability <= 0:
        return 0.0

    return (1 + factor * elapsed_days / stability) ** (-w20)


def update_stability(
    old_stability: float,
    difficulty: float,
    retention: float,
) -> float:
    """Update stability after successful retrieval (FSRS-6 simplified)."""
    # Higher retention at recall = bigger stability boost
    # Lower difficulty = bigger stability boost
    boost = 1.0 + (1.0 - difficulty) * (1.0 + retention)
    return old_stability * boost


# =============================================================================
# Embedding (simple fallback - real system would use fastembed)
# =============================================================================

def simple_embedding(text: str, dim: int = 128) -> list[float]:
    """Generate a simple hash-based pseudo-embedding.

    This is a fallback - production would use fastembed/nomic.
    The embedding captures some lexical features via hashing.
    """
    # Normalize text
    words = text.lower().split()

    # Initialize embedding
    emb = np.zeros(dim, dtype=np.float32)

    for word in words:
        # Hash word to get indices
        h = hashlib.md5(word.encode()).hexdigest()
        for i in range(0, len(h), 2):
            idx = int(h[i:i+2], 16) % dim
            sign = 1 if int(h[i], 16) < 8 else -1
            emb[idx] += sign * 0.1

    # Normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# =============================================================================
# Memory Store
# =============================================================================

class MemoryStore:
    """SQLite-backed memory store with hybrid search."""

    def __init__(self, db_path: Path = MEMORY_DB):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database with FTS5 for keyword search."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding TEXT,
                    storage_strength REAL DEFAULT 1.0,
                    retrieval_strength REAL DEFAULT 1.0,
                    stability REAL DEFAULT 1.0,
                    difficulty REAL DEFAULT 0.3,
                    created_at TEXT,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 1,
                    tags TEXT,
                    source TEXT,
                    metadata TEXT
                )
            """)

            # FTS5 for keyword search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    id,
                    content,
                    tags,
                    content=memories,
                    content_rowid=rowid
                )
            """)

            # Trigger to keep FTS in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, id, content, tags)
                    VALUES (new.rowid, new.id, new.content, new.tags);
                END
            """)

            conn.commit()

    def store(self, memory: Memory) -> str:
        """Store a memory, returning its ID."""
        # Generate embedding if not provided
        if memory.embedding is None:
            memory.embedding = simple_embedding(memory.content)

        # Generate ID from content hash if not provided
        if not memory.id:
            memory.id = hashlib.sha256(memory.content.encode()).hexdigest()[:16]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, content, embedding, storage_strength, retrieval_strength,
                 stability, difficulty, created_at, last_accessed, access_count,
                 tags, source, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.content,
                json.dumps(memory.embedding),
                memory.storage_strength,
                memory.retrieval_strength,
                memory.stability,
                memory.difficulty,
                memory.created_at.isoformat(),
                memory.last_accessed.isoformat(),
                memory.access_count,
                json.dumps(memory.tags),
                memory.source,
                json.dumps(memory.metadata),
            ))
            conn.commit()

        return memory.id

    def get(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_memory(row)

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            id=row["id"],
            content=row["content"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            storage_strength=row["storage_strength"],
            retrieval_strength=row["retrieval_strength"],
            stability=row["stability"],
            difficulty=row["difficulty"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            source=row["source"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        min_accessibility: float = 0.1,
        semantic_weight: float = 0.5,
    ) -> list[tuple[Memory, float]]:
        """Hybrid search combining keyword and semantic matching.

        Uses Reciprocal Rank Fusion (RRF) to combine rankings.

        Args:
            query: Search query
            limit: Max results
            min_accessibility: Filter out low-accessibility memories
            semantic_weight: Balance between semantic (1.0) and keyword (0.0)

        Returns:
            List of (Memory, score) tuples
        """
        query_embedding = simple_embedding(query)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get all memories (for a production system, this would be optimized)
            rows = conn.execute("SELECT * FROM memories").fetchall()

        if not rows:
            return []

        # Score each memory
        scored: list[tuple[Memory, float, float]] = []  # (memory, keyword_score, semantic_score)

        for row in rows:
            memory = self._row_to_memory(row)

            # Skip low-accessibility memories
            if memory.accessibility < min_accessibility:
                continue

            # Keyword score (simple word overlap)
            query_words = set(query.lower().split())
            content_words = set(memory.content.lower().split())
            tag_words = set(" ".join(memory.tags).lower().split())
            all_words = content_words | tag_words

            if query_words:
                keyword_score = len(query_words & all_words) / len(query_words)
            else:
                keyword_score = 0.0

            # Semantic score
            if memory.embedding:
                semantic_score = max(0, cosine_similarity(query_embedding, memory.embedding))
            else:
                semantic_score = 0.0

            scored.append((memory, keyword_score, semantic_score))

        if not scored:
            return []

        # RRF fusion
        k = 60  # RRF constant

        # Rank by keyword score
        keyword_ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        keyword_ranks = {m.id: i + 1 for i, (m, _, _) in enumerate(keyword_ranked)}

        # Rank by semantic score
        semantic_ranked = sorted(scored, key=lambda x: x[2], reverse=True)
        semantic_ranks = {m.id: i + 1 for i, (m, _, _) in enumerate(semantic_ranked)}

        # Compute RRF scores
        results: list[tuple[Memory, float]] = []
        for memory, kw_score, sem_score in scored:
            kw_rrf = (1 - semantic_weight) / (k + keyword_ranks[memory.id])
            sem_rrf = semantic_weight / (k + semantic_ranks[memory.id])

            # Also factor in accessibility
            rrf_score = (kw_rrf + sem_rrf) * memory.accessibility

            results.append((memory, rrf_score))

        # Sort by final score
        results.sort(key=lambda x: x[1], reverse=True)

        # Access the retrieved memories (testing effect)
        for memory, _ in results[:limit]:
            memory.access()
            self._update_access(memory)

        return results[:limit]

    def _update_access(self, memory: Memory) -> None:
        """Update memory access stats in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE memories SET
                    storage_strength = ?,
                    retrieval_strength = ?,
                    stability = ?,
                    last_accessed = ?,
                    access_count = ?
                WHERE id = ?
            """, (
                memory.storage_strength,
                memory.retrieval_strength,
                memory.stability,
                memory.last_accessed.isoformat(),
                memory.access_count,
                memory.id,
            ))
            conn.commit()

    def smart_ingest(
        self,
        content: str,
        tags: list[str] | None = None,
        source: str = "",
        metadata: dict | None = None,
    ) -> tuple[str, str]:
        """Intelligently ingest content with duplicate detection.

        Based on similarity:
        - > 0.92: REINFORCE existing (just strengthen)
        - > 0.75: UPDATE existing (merge information)
        - < 0.75: CREATE new memory

        Returns:
            Tuple of (action, memory_id)
        """
        query_embedding = simple_embedding(content)

        # Find similar memories
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM memories").fetchall()

        best_match: tuple[Memory, float] | None = None

        for row in rows:
            memory = self._row_to_memory(row)
            if memory.embedding:
                sim = cosine_similarity(query_embedding, memory.embedding)
                if best_match is None or sim > best_match[1]:
                    best_match = (memory, sim)

        if best_match and best_match[1] > 0.92:
            # REINFORCE - just strengthen existing
            memory = best_match[0]
            memory.access()
            memory.storage_strength += 0.2
            self._update_access(memory)
            return ("REINFORCE", memory.id)

        elif best_match and best_match[1] > 0.75:
            # UPDATE - merge with existing
            memory = best_match[0]
            memory.content = f"{memory.content}\n\n[Updated]: {content}"
            memory.embedding = simple_embedding(memory.content)
            memory.access()
            if tags:
                memory.tags = list(set(memory.tags + tags))
            self.store(memory)
            return ("UPDATE", memory.id)

        else:
            # CREATE - new memory
            memory = Memory(
                id="",
                content=content,
                embedding=query_embedding,
                tags=tags or [],
                source=source,
                metadata=metadata or {},
            )
            memory_id = self.store(memory)
            return ("CREATE", memory_id)

    def promote(self, memory_id: str) -> bool:
        """Mark memory as helpful (strengthens it)."""
        memory = self.get(memory_id)
        if memory is None:
            return False

        memory.storage_strength += 0.5
        memory.retrieval_strength = min(1.0, memory.retrieval_strength + 0.3)
        memory.difficulty = max(0.1, memory.difficulty - 0.1)  # Easier to remember
        self.store(memory)
        return True

    def demote(self, memory_id: str) -> bool:
        """Mark memory as wrong/unhelpful (weakens it)."""
        memory = self.get(memory_id)
        if memory is None:
            return False

        memory.retrieval_strength *= 0.5
        memory.difficulty = min(0.9, memory.difficulty + 0.1)  # Harder to recall
        self.store(memory)
        return True

    def decay_all(self, hours: float = 1.0) -> int:
        """Apply time-based decay to all memories."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM memories").fetchall()

        count = 0
        for row in rows:
            memory = self._row_to_memory(row)
            memory.decay(hours)
            self.store(memory)
            count += 1

        return count

    def stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM memories").fetchall()

        if total == 0:
            return {"total": 0, "by_state": {}}

        by_state = {state.value: 0 for state in MemoryState}
        total_accessibility = 0.0

        for row in rows:
            memory = self._row_to_memory(row)
            by_state[memory.get_state().value] += 1
            total_accessibility += memory.accessibility

        return {
            "total": total,
            "by_state": by_state,
            "avg_accessibility": total_accessibility / total,
        }


# =============================================================================
# Global memory store instance
# =============================================================================

_memory_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Get the global memory store instance."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store
