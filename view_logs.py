import sqlite3
import time

def view_premise_logs():
    db_path = "database/visitors.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT visitor_id, path_taken, entry_time, exit_time, total_duration FROM premise_logs ORDER BY entry_time DESC")
        logs = cursor.fetchall()
        
        if not logs:
            print("\n[INFO] No access logs found yet. Walk in front of the camera, then leave the frame for 15 seconds to generate a log.")
            return

        print("\n" + "="*85)
        print(f"{'VISITOR ID':<20} | {'PATH TAKEN':<15} | {'ENTRY TIME':<20} | {'DURATION (Sec)'}")
        print("="*85)
        
        for log in logs:
            v_id = log[0]
            path = log[1]
            entry = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log[2]))
            duration = round(log[4], 1)
            print(f"{v_id:<20} | {path:<15} | {entry:<20} | {duration}s")
            
        print("="*85 + "\n")
        
    except sqlite3.Error as e:
        print(f"[ERROR] Could not read database: {e}")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    view_premise_logs()