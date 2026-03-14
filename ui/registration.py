import tkinter as tk
from tkinter import messagebox
import cv2

class RegistrationUI:
    def __init__(self, db_handler, state_manager, reid_matcher):
        self.db = db_handler
        self.state_manager = state_manager
        self.reid_matcher = reid_matcher
        
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("Visitor Registration System")
        self.root.geometry("400x350")

        # --- UI ELEMENTS ---
        tk.Label(self.root, text="Visitor ID:").grid(row=0, column=0, pady=10, padx=10)
        self.entry_id = tk.Entry(self.root)
        self.entry_id.grid(row=0, column=1)

        tk.Label(self.root, text="Name:").grid(row=1, column=0, pady=10)
        self.entry_name = tk.Entry(self.root)
        self.entry_name.grid(row=1, column=1)

        tk.Label(self.root, text="Visiting Location:").grid(row=2, column=0, pady=10)
        self.entry_loc = tk.Entry(self.root)
        self.entry_loc.grid(row=2, column=1)

        tk.Label(self.root, text="Vehicle Number:").grid(row=3, column=0, pady=10)
        self.entry_veh = tk.Entry(self.root)
        self.entry_veh.grid(row=3, column=1)

        tk.Label(self.root, text="Time Limit (seconds):").grid(row=4, column=0, pady=10)
        self.entry_time = tk.Entry(self.root)
        self.entry_time.insert(0, "300") # Default to 300 seconds (5 mins)
        self.entry_time.grid(row=4, column=1)

        # Register Button
        self.btn_register = tk.Button(self.root, text="Capture Face & Register", bg="green", fg="white", command=self.register_action)
        self.btn_register.grid(row=5, column=0, columnspan=2, pady=20)

    def register_action(self):
        """Grabs the data, takes a webcam picture, and saves it all."""
        v_id = self.entry_id.get()
        name = self.entry_name.get()
        loc = self.entry_loc.get()
        veh = self.entry_veh.get()
        
        try:
            threshold = int(self.entry_time.get())
        except ValueError:
            messagebox.showerror("Error", "Time limit must be a number.")
            return

        if not v_id or not name:
            messagebox.showerror("Error", "ID and Name are required!")
            return

        # 1. Capture Face from Registration Webcam (Camera 0)
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Camera Error", "Could not connect to registration webcam.")
            return

        # For this template, we assume the person is taking up the whole frame.
        # In a real setup, you'd run YOLO here to find their exact bounding box first.
        height, width, _ = frame.shape
        bbox = [0, 0, width, height] 

        # 2. Save Visual Signature to AI Brain
        self.reid_matcher.register_visitor_visuals(v_id, frame, bbox)

        # 3. Save to Permanent Database
        self.db.register_visitor(v_id, name, loc, veh, threshold)

        # 4. Add to Active State Manager
        self.state_manager.add_visitor(v_id, threshold)

        messagebox.showinfo("Success", f"Visitor {name} registered successfully!")
        self.root.destroy() # Close UI after successful registration

    def run(self):
        """Starts the Tkinter loop."""
        self.root.mainloop()