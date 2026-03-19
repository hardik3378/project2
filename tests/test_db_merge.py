import numpy as np

from database.db_handler import DBHandler


def test_merge_preserves_logs_and_canonical_signature(tmp_path):
    db = DBHandler(str(tmp_path / "visitors.db"))

    db.register_visitor("Stranger_1", "Unverified", "cam_1", "N/A", 300)
    db.register_visitor("Stranger_2", "Unverified", "cam_1", "N/A", 300)

    sig_keep = np.ones((1, 576), dtype=np.float32)
    sig_merge = np.full((1, 576), 2.0, dtype=np.float32)
    combined = np.vstack([sig_keep, sig_merge])

    db.save_signature("Stranger_1", sig_keep)
    db.save_signature("Stranger_2", sig_merge)

    db.save_premise_log("Stranger_2", "cam_1", 1.0, 5.0, 4.0)
    assert db.merge_visitors("Stranger_1", "Stranger_2", combined) is True

    cursor = db.conn.cursor()
    cursor.execute("SELECT visitor_id FROM premise_logs")
    assert cursor.fetchall() == [("Stranger_1",)]

    loaded = db.load_all_signatures()
    assert "Stranger_1" in loaded
    assert "Stranger_2" not in loaded
    assert loaded["Stranger_1"].shape == (2, 576)
    np.testing.assert_allclose(loaded["Stranger_1"], combined)

    cursor.execute(
        "SELECT embedding_dim, embedding_model, embedding_version FROM signatures WHERE visitor_id = ?",
        ("Stranger_1",),
    )
    assert cursor.fetchone() == (576, "mobilenet_v3_small", "imagenet_default")
    db.close()


def test_load_all_signatures_uses_embedding_dimension_metadata(tmp_path):
    db = DBHandler(str(tmp_path / "visitors.db"))
    db.register_visitor("Person_1", "Verified", "cam_1", "Alice", 300)

    custom = np.arange(12, dtype=np.float32).reshape(3, 4)
    db.conn.cursor().execute(
        """
        INSERT INTO signatures (visitor_id, signature_data, embedding_dim, embedding_model, embedding_version)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("Person_1", custom.tobytes(), 4, "tiny-model", "v2"),
    )
    db.conn.commit()

    loaded = db.load_all_signatures()
    assert loaded["Person_1"].shape == (3, 4)
    np.testing.assert_allclose(loaded["Person_1"], custom)
    db.close()
