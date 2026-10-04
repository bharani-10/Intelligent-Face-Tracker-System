import numpy as np

from app.visitor_manager import VisitorManager


class FakeImageLogger:
    def __init__(self):
        self.saved = []

    def save_event_image(self, crop, visitor_id, event_type, timestamp):
        self.saved.append((visitor_id, event_type, timestamp, crop))
        return f"logs/{event_type.lower()}/{visitor_id}_{timestamp}.jpg"


class FakeDatabase:
    def __init__(self):
        self.visitors = {}
        self.events = []

    def get_visitor_embeddings(self):
        return [(visitor_id, record["embedding"]) for visitor_id, record in self.visitors.items()]

    def get_unique_visitor_count(self):
        return len(self.visitors)

    def register_visitor(self, visitor_id, embedding, timestamp):
        self.visitors[visitor_id] = {"embedding": embedding, "timestamp": timestamp}

    def update_last_seen(self, visitor_id, timestamp):
        self.visitors[visitor_id]["last_seen"] = timestamp

    def log_event(self, visitor_id, event_type, timestamp, image_path, track_id):
        self.events.append({"visitor_id": visitor_id, "event_type": event_type})

    def get_event_count(self, event_type=None):
        if event_type is None:
            return len(self.events)
        return sum(event["event_type"] == event_type for event in self.events)


def test_two_visits_have_one_entry_and_exit_and_one_visitor(tmp_path):
    database = FakeDatabase()
    image_logger = FakeImageLogger()
    manager = VisitorManager(database, image_logger, similarity_threshold=0.8, missing_timeout_frames=1)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    crop = np.ones((10, 10, 3), dtype=np.uint8)

    manager.process_frame(1, "2026-10-04T10:00:01", [{"track_id": 7}], {7: crop}, {7: embedding})
    manager.process_frame(2, "2026-10-04T10:00:02", [{"track_id": 7}], {}, {})
    manager.process_frame(3, "2026-10-04T10:00:03", [], {}, {})
    manager.process_frame(4, "2026-10-04T10:00:04", [{"track_id": 7}], {}, {})
    manager.process_frame(5, "2026-10-04T10:00:05", [], {}, {})
    manager.process_frame(6, "2026-10-04T10:00:06", [], {}, {})
    manager.process_frame(7, "2026-10-04T10:00:07", [{"track_id": 8}], {8: crop}, {8: embedding})
    manager.process_frame(8, "2026-10-04T10:00:08", [], {}, {})
    manager.process_frame(9, "2026-10-04T10:00:09", [], {}, {})

    events = database.events
    assert [event["event_type"] for event in events] == ["ENTRY", "EXIT", "ENTRY", "EXIT"]
    assert {event["visitor_id"] for event in events} == {"VISITOR_0001"}
    assert database.get_unique_visitor_count() == 1
    assert database.get_event_count("ENTRY") == 2
    assert database.get_event_count("EXIT") == 2