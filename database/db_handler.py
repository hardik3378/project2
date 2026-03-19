import os
import sqlite3

import numpy as np


class DBHandler:
    DEFAULT_EMBEDDING_DIM = 576
    DEFAULT_EMBEDDING_MODEL = "mobilenet_v3_small"
    DEFAULT_MODEL_VERSION = "imagenet_default"

    def __init__(self, db_path="database/visitors.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _column_exists(self, table_name, column_name):
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return any(row[1] == column_name for row in cursor.fetchall())

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS visitors (
                id TEXT PRIMARY KEY,
                auth_level TEXT,
                allowed_zone TEXT,
                known_name TEXT,
                threshold_time INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS signatures (
                visitor_id TEXT PRIMARY KEY,
                signature_data BLOB,
                embedding_dim INTEGER NOT NULL DEFAULT 576,
                embedding_model TEXT NOT NULL DEFAULT 'mobilenet_v3_small',
                embedding_version TEXT NOT NULL DEFAULT 'imagenet_default',
                FOREIGN KEY(visitor_id) REFERENCES visitors(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS premise_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id TEXT,
                path_taken TEXT,
                entry_time REAL,
                exit_time REAL,
                total_duration REAL,
                FOREIGN KEY(visitor_id) REFERENCES visitors(id)
            )
            """
        )

        # Forward-compatible migration for older DBs that had only signature_data.
        if not self._column_exists("signatures", "embedding_dim"):
            cursor.execute("ALTER TABLE signatures ADD COLUMN embedding_dim INTEGER NOT NULL DEFAULT 576")
        if not self._column_exists("signatures", "embedding_model"):
            cursor.execute("ALTER TABLE signatures ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'mobilenet_v3_small'")
        if not self._column_exists("signatures", "embedding_version"):
            cursor.execute("ALTER TABLE signatures ADD COLUMN embedding_version TEXT NOT NULL DEFAULT 'imagenet_default'")

        self.conn.commit()

    def _normalize_signatures(self, signature_array):
        if signature_array.ndim == 1:
            return np.array([signature_array], dtype=np.float32)
        return np.asarray(signature_array, dtype=np.float32)

    def merge_visitors(self, keep_id, merge_id, combined_signatures):
        """Transfers logs to keep_id, saves canonical signature rows, and deletes duplicate visitor."""
        cursor = self.conn.cursor()
        try:
            combined_signatures = self._normalize_signatures(combined_signatures)

            cursor.execute("UPDATE premise_logs SET visitor_id = ? WHERE visitor_id = ?", (keep_id, merge_id))

            cursor.execute(
                """
                UPDATE signatures
                SET signature_data = ?, embedding_dim = ?, embedding_model = ?, embedding_version = ?
                WHERE visitor_id = ?
                """,
                (
                    combined_signatures.tobytes(),
                    int(combined_signatures.shape[1]),
                    self.DEFAULT_EMBEDDING_MODEL,
                    self.DEFAULT_MODEL_VERSION,
                    keep_id,
                ),
            )

            cursor.execute("DELETE FROM signatures WHERE visitor_id = ?", (merge_id,))
            cursor.execute("DELETE FROM visitors WHERE id = ?", (merge_id,))

            self.conn.commit()
            return True
        except Exception as e:
            print(f"[DB ERROR] Merge failed: {e}")
            return False

    def save_premise_log(self, visitor_id, path_taken, entry_time, exit_time, duration):
        self.conn.cursor().execute(
            "INSERT INTO premise_logs (visitor_id, path_taken, entry_time, exit_time, total_duration) VALUES (?, ?, ?, ?, ?)",
            (visitor_id, path_taken, entry_time, exit_time, duration),
        )
        self.conn.commit()

    def register_visitor(self, person_id, auth_level, zone, name, time_limit):
        self.conn.cursor().execute(
            "INSERT OR IGNORE INTO visitors (id, auth_level, allowed_zone, known_name, threshold_time) VALUES (?, ?, ?, ?, ?)",
            (person_id, auth_level, zone, name, time_limit),
        )
        self.conn.commit()

    def save_signature(self, person_id, signature_array):
        signature_array = self._normalize_signatures(signature_array)
        self.conn.cursor().execute(
            """
            INSERT OR IGNORE INTO signatures
            (visitor_id, signature_data, embedding_dim, embedding_model, embedding_version)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                person_id,
                signature_array.tobytes(),
                int(signature_array.shape[1]),
                self.DEFAULT_EMBEDDING_MODEL,
                self.DEFAULT_MODEL_VERSION,
            ),
        )
        self.conn.commit()

    def update_signature(self, person_id, signature_array):
        signature_array = self._normalize_signatures(signature_array)
        self.conn.cursor().execute(
            """
            UPDATE signatures
            SET signature_data = ?, embedding_dim = ?, embedding_model = ?, embedding_version = ?
            WHERE visitor_id = ?
            """,
            (
                signature_array.tobytes(),
                int(signature_array.shape[1]),
                self.DEFAULT_EMBEDDING_MODEL,
                self.DEFAULT_MODEL_VERSION,
                person_id,
            ),
        )
        self.conn.commit()

    def load_all_signatures(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT visitor_id, signature_data, embedding_dim FROM signatures")
        sigs = {}
        for visitor_id, signature_blob, embedding_dim in cursor.fetchall():
            dim = int(embedding_dim or self.DEFAULT_EMBEDDING_DIM)
            sigs[visitor_id] = np.frombuffer(signature_blob, dtype=np.float32).reshape(-1, dim)
        return sigs

    def reassign_visitor(self, old_id, new_id, name, auth_level):
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT id, allowed_zone, threshold_time FROM visitors WHERE id = ?", (old_id,))
            row = cursor.fetchone()
            if not row:
                return False

            _, allowed_zone, threshold_time = row
            cursor.execute(
                """
                INSERT OR REPLACE INTO visitors (id, auth_level, allowed_zone, known_name, threshold_time)
                VALUES (?, ?, ?, ?, ?)
                """,
                (new_id, auth_level, allowed_zone, name, threshold_time),
            )
            cursor.execute("UPDATE signatures SET visitor_id = ? WHERE visitor_id = ?", (new_id, old_id))
            cursor.execute("UPDATE premise_logs SET visitor_id = ? WHERE visitor_id = ?", (new_id, old_id))
            cursor.execute("DELETE FROM visitors WHERE id = ?", (old_id,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[DB ERROR] Reassign failed: {e}")
            self.conn.rollback()
            return False

    def get_all_logs(self):
        return self.conn.cursor().execute(
            "SELECT visitor_id, path_taken, entry_time, exit_time, total_duration FROM premise_logs ORDER BY entry_time DESC"
        ).fetchall()

    def close(self):
        self.conn.close()
