class EventEngine:
    def __init__(self):
        self.active_tracks = set()
        self.events = []

    def update(self, detections):
        current_tracks = {d["track_id"] for d in detections if d["track_id"] != -1}
        
        # Entries
        entries = current_tracks - self.active_tracks
        for tid in entries:
            label = next(d["label"] for d in detections if d["track_id"] == tid)
            self.events.append({"type": "Entry", "msg": f"{label} (ID:{tid}) entered scene"})
            
        # Exits
        exits = self.active_tracks - current_tracks
        for tid in exits:
            self.events.append({"type": "Exit", "msg": f"Object ID:{tid} left scene"})
            
        self.active_tracks = current_tracks
        
        # Keep only last 50 events
        if len(self.events) > 50: self.events = self.events[-50:]
        return self.events

_engine = EventEngine()
def get_event_engine():
    return _engine
