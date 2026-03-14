import time
import threading

class AlertSystem:
    def __init__(self, state_manager, alert_check_interval=5, **kwargs):
        self.state_manager = state_manager
        self.check_interval = alert_check_interval
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("[ALERT SYSTEM] Background monitoring started.")

    def stop(self):
        self.running = False

    def _monitor_loop(self):
        while self.running:
            current_time = time.time()
            active_visitors = list(self.state_manager.get_all_active().items())
            for visitor_id, data in active_visitors:
                elapsed_time = current_time - data['entry_time']
                if elapsed_time > data['threshold']:
                    print(f"!!! [ALERT] Visitor {visitor_id} time exceeded !!!")
            time.sleep(self.check_interval)