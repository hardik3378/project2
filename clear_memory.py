import os
import glob

def clear_ai_memory():
    print("=========================================")
    print("        AI MEMORY WIPE UTILITY           ")
    print("=========================================")
    print("WARNING: This will permanently delete:")
    print(" 1. All known faces and merged identities.")
    print(" 2. All visitor access logs and timelines.")
    print(" 3. All saved images of strangers.")
    print("=========================================")
    
    confirm = input("Are you sure you want to wipe the AI's memory? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("\n[CANCELLED] Memory wipe aborted. Your data is safe.")
        return

    print("\n[INFO] Initiating memory wipe...")

    # 1. Delete the SQLite Database
    db_path = os.path.join("database", "visitors.db")
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"[SUCCESS] Deleted database: {db_path}")
        except Exception as e:
            print(f"[ERROR] Could not delete database. Is main.py still running? ({e})")
    else:
        print("[INFO] Database file not found (already empty).")

    # 2. Delete the saved stranger images
    log_dir = os.path.join("logs", "strangers")
    if os.path.exists(log_dir):
        images = glob.glob(os.path.join(log_dir, "*.jpg"))
        if images:
            count = 0
            for img in images:
                try:
                    os.remove(img)
                    count += 1
                except Exception as e:
                    print(f"[ERROR] Could not delete {img}: {e}")
            print(f"[SUCCESS] Deleted {count} saved stranger images from '{log_dir}'.")
        else:
            print("[INFO] No stranger images found to delete.")
    else:
        print("[INFO] Image logs directory not found (already empty).")

    print("\n[COMPLETE] AI Memory has been completely wiped!")
    print("The system will build a fresh database the next time you run main.py.\n")

if __name__ == "__main__":
    clear_ai_memory()