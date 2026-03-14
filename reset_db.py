import yaml
from database.db_handler import DBHandler

def reset_database():
    print("Connecting to PostgreSQL...")
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        db = DBHandler(**config['database'])
        cursor = db.conn.cursor()
        
        # TRUNCATE clears the table completely and resets any auto-incrementing IDs
        # CASCADE ensures any linked tables (like signatures) are also cleared
        cursor.execute("TRUNCATE TABLE visitors CASCADE;")
        db.conn.commit()
        
        print("\n[SUCCESS] Database completely cleared!")
        print("All old RGB signatures and IDs have been wiped.")
        print("You are ready to run main.py with the new HSV logic.")
        
    except Exception as e:
        print(f"\n[ERROR] Could not clear database: {e}")
    finally:
        if 'db' in locals():
            db.close()

if __name__ == "__main__":
    reset_database()