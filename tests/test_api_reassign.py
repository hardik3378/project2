import pytest

pytest.importorskip("flask")

from api_server import create_app


class FakeDB:
    def __init__(self, should_succeed=True):
        self.should_succeed = should_succeed
        self.calls = []

    def reassign_visitor(self, old_id, new_id, name, auth_level):
        self.calls.append((old_id, new_id, name, auth_level))
        return self.should_succeed


class FakeMatcher:
    def __init__(self):
        self.calls = []

    def update_id_in_memory(self, old_id, new_id):
        self.calls.append((old_id, new_id))


def test_reassign_validation_error_missing_ids(tmp_path):
    db = FakeDB()
    matcher = FakeMatcher()
    app = create_app(db_handler=db, matcher=matcher)

    client = app.test_client()
    resp = client.post("/api/reassign", json={"old_id": "Stranger_1"})

    assert resp.status_code == 400
    assert db.calls == []
    assert matcher.calls == []


def test_reassign_success_path(tmp_path):
    db = FakeDB(should_succeed=True)
    matcher = FakeMatcher()
    app = create_app(db_handler=db, matcher=matcher)

    client = app.test_client()
    payload = {
        "old_id": "Stranger_1",
        "new_id": "EMP_001",
        "name": "John Doe",
        "auth_level": "Employee",
    }
    resp = client.post("/api/reassign", json=payload)

    assert resp.status_code == 200
    assert db.calls == [("Stranger_1", "EMP_001", "John Doe", "Employee")]
    assert matcher.calls == [("Stranger_1", "EMP_001")]
