import sqlite3
import numpy as np
import os

class DBHandler:
    def __init__(self, db_path="database/visitors.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS visitors (id TEXT PRIMARY KEY, auth_level TEXT, allowed_zone TEXT, known_name TEXT, threshold_time INTEGER)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS signatures (visitor_id TEXT PRIMARY KEY, signature_data BLOB, FOREIGN KEY(visitor_id) REFERENCES visitors(id))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS premise_logs (log_id INTEGER PRIMARY KEY AUTOINCREMENT, visitor_id TEXT, path_taken TEXT, entry_time REAL, exit_time REAL, total_duration REAL, FOREIGN KEY(visitor_id) REFERENCES visitors(id))''')
        self.conn.commit()

    # --- NEW: Database Merger Logic ---
    def merge_visitors(self, keep_id, merge_id, combined_signatures):
        """Transfers logs to the main ID, saves the combined memory, and deletes the duplicate."""
        cursor = self.conn.cursor()
        try:
            # 1. Transfer premise logs to the older/kept ID
            cursor.execute("UPDATE premise_logs SET visitor_id = ? WHERE visitor_id = ?", (keep_id, merge_id))
            
            # 2. Save the newly fused 20-pattern memory
            if combined_signatures.ndim == 1: combined_signatures = np.array([combined_signatures])
            cursor.execute("UPDATE signatures SET signature_data = ? WHERE visitor_id = ?", (combined_signatures.tobytes(), keep_id))
            
            # 3. Erase the duplicate ghost ID
            cursor.execute("DELETE FROM signatures WHERE visitor_id = ?", (merge_id,))
            cursor.execute("DELETE FROM visitors WHERE id = ?", (merge_id,))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[DB ERROR] Merge failed: {e}")
            return False

    def save_premise_log(self, visitor_id, path_taken, entry_time, exit_time, duration):
        self.conn.cursor().execute("INSERT INTO premise_logs (visitor_id, path_taken, entry_time, exit_time, total_duration) VALUES (?, ?, ?, ?, ?)", (visitor_id, path_taken, entry_time, exit_time, duration))
        self.conn.commit()

    def register_visitor(self, person_id, auth_level, zone, name, time_limit):
        self.conn.cursor().execute("INSERT OR IGNORE INTO visitors (id, auth_level, allowed_zone, known_name, threshold_time) VALUES (?, ?, ?, ?, ?)", (person_id, auth_level, zone, name, time_limit))
        self.conn.commit()

    def save_signature(self, person_id, signature_array):
        if signature_array.ndim == 1: signature_array = np.array([signature_array])
        self.conn.cursor().execute("INSERT OR IGNORE INTO signatures (visitor_id, signature_data) VALUES (?, ?)", (person_id, signature_array.tobytes()))
        self.conn.commit()

    def update_signature(self, person_id, signature_array):
        if signature_array.ndim == 1: signature_array = np.array([signature_array])
        self.conn.cursor().execute("UPDATE signatures SET signature_data = ? WHERE visitor_id = ?", (signature_array.tobytes(), person_id))
        self.conn.commit()

    def load_all_signatures(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT visitor_id, signature_data FROM signatures")
        sigs = {}
        for row in cursor.fetchall():
            sigs[row[0]] = np.frombuffer(row[1], dtype=np.float32).reshape(-1, 576) 
        return sigs

    def get_all_logs(self):
        return self.conn.cursor().execute("SELECT visitor_id, path_taken, entry_time, exit_time, total_duration FROM premise_logs ORDER BY entry_time DESC").fetchall()

    def close(self): self.conn.close()