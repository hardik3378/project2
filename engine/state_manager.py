import time

class StateManager:
    def __init__(self):
        self.active_visitors = {}
        self.active_strangers = {}

    def add_visitor(self, v_id, threshold):
        self.active_visitors[v_id] = {
            'entry_time': time.time(),
            'threshold': threshold,
            'last_cam': 'Cam_1',
            'type': 'Registered'
        }

    def add_stranger(self, s_id):
        if s_id not in self.active_strangers:
            self.active_strangers[s_id] = {
                'entry_time': time.time(),
                'last_cam': 'Cam_1',
                'type': 'Stranger'
            }

    def update_location(self, entity_id, cam_id, is_stranger=False):
        target_dict = self.active_strangers if is_stranger else self.active_visitors
        if entity_id in target_dict:
            target_dict[entity_id]['last_cam'] = cam_id

    def remove_entity(self, entity_id, is_stranger=False):
        target_dict = self.active_strangers if is_stranger else self.active_visitors
        if entity_id in target_dict:
            duration = time.time() - target_dict[entity_id]['entry_time']
            del target_dict[entity_id]
            return duration
        return None

    def get_total_active_count(self):
        return len(self.active_visitors) + len(self.active_strangers)

    def get_all_active(self):
        # Combines both for the dashboard logic
        return {**self.active_visitors, **self.active_strangers}