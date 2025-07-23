"""
Global semaphore for controlling concurrent agent spawning
Prevents runaway agent creation and memory blow-up
"""

import sqlite3
import contextlib
import time
import threading
from pathlib import Path
from typing import Optional

# Global lock for thread safety
_lock = threading.Lock()


class GlobalSemaphore:
    """
    Global semaphore using SQLite for cross-process coordination
    Prevents unlimited agent spawning across all instances
    """
    
    def __init__(self, db_path: str = "./runtime_meta.db", cleanup_interval: int = 3600):
        self.db_path = Path(db_path)
        self.cleanup_interval = cleanup_interval  # Cleanup entries older than this (seconds)
        self._init_db()
    
    def _init_db(self):
        """Initialize the semaphore database"""
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semaphore (
                    tag TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    process_id TEXT,
                    agent_type TEXT
                )
            """)
            conn.commit()
    
    def try_acquire(self, tag: str, max_active: int = 10, agent_type: str = "generic") -> bool:
        """
        Try to acquire a semaphore slot
        
        Args:
            tag: Unique identifier for this agent
            max_active: Maximum number of active agents allowed
            agent_type: Type of agent being spawned
            
        Returns:
            True if slot acquired, False if limit reached
        """
        with _lock:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                # Cleanup stale entries
                cutoff_time = time.time() - self.cleanup_interval
                conn.execute("DELETE FROM semaphore WHERE ts < ?", (cutoff_time,))
                
                # Check current count
                cursor = conn.execute("SELECT COUNT(*) FROM semaphore")
                current_count = cursor.fetchone()[0]
                
                if current_count >= max_active:
                    return False
                
                # Acquire slot
                try:
                    conn.execute(
                        "INSERT INTO semaphore (tag, ts, process_id, agent_type) VALUES (?, ?, ?, ?)",
                        (tag, time.time(), str(threading.get_ident()), agent_type)
                    )
                    conn.commit()
                    return True
                except sqlite3.IntegrityError:
                    # Tag already exists
                    return False
    
    def release(self, tag: str):
        """
        Release a semaphore slot
        
        Args:
            tag: Unique identifier for the agent to release
        """
        with _lock:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute("DELETE FROM semaphore WHERE tag = ?", (tag,))
                conn.commit()
    
    def get_active_count(self) -> int:
        """Get current number of active agents"""
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            # Cleanup first
            cutoff_time = time.time() - self.cleanup_interval
            conn.execute("DELETE FROM semaphore WHERE ts < ?", (cutoff_time,))
            conn.commit()
            
            cursor = conn.execute("SELECT COUNT(*) FROM semaphore")
            return cursor.fetchone()[0]
    
    def get_active_agents(self) -> list:
        """Get list of currently active agents"""
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            # Cleanup first
            cutoff_time = time.time() - self.cleanup_interval
            conn.execute("DELETE FROM semaphore WHERE ts < ?", (cutoff_time,))
            
            cursor = conn.execute("SELECT tag, agent_type, ts FROM semaphore ORDER BY ts")
            return [{"tag": row[0], "agent_type": row[1], "started": row[2]} for row in cursor.fetchall()]
    
    def force_cleanup(self):
        """Force cleanup of all semaphore entries"""
        with _lock:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute("DELETE FROM semaphore")
                conn.commit()


# Global instance
global_semaphore = GlobalSemaphore()


# Context manager for easy usage
class SemaphoreGuard:
    """Context manager for semaphore usage"""
    
    def __init__(self, tag: str, max_active: int = 10, agent_type: str = "generic"):
        self.tag = tag
        self.max_active = max_active
        self.agent_type = agent_type
        self.acquired = False
    
    def __enter__(self):
        self.acquired = global_semaphore.try_acquire(self.tag, self.max_active, self.agent_type)
        if not self.acquired:
            raise RuntimeError(f"Could not acquire semaphore slot (max {self.max_active} active)")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            global_semaphore.release(self.tag)


# Convenience functions
def try_acquire(tag: str, max_active: int = 10, agent_type: str = "generic") -> bool:
    """Try to acquire a semaphore slot"""
    return global_semaphore.try_acquire(tag, max_active, agent_type)


def release(tag: str):
    """Release a semaphore slot"""
    global_semaphore.release(tag)


def get_active_count() -> int:
    """Get current number of active agents"""
    return global_semaphore.get_active_count()


def get_active_agents() -> list:
    """Get list of currently active agents"""
    return global_semaphore.get_active_agents()


# Example usage:
# with SemaphoreGuard("my_agent_123", max_active=5, agent_type="creative_analyst"):
#     # Agent work here
#     pass