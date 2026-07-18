"""Contract tests on fixture JSON (no network)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schema import (  # noqa: E402
    SchemaError,
    session_looks_authenticated,
    validate_assignments,
    validate_sessions,
)


class SchemaTests(unittest.TestCase):
    def test_sessions_fixture(self) -> None:
        data = json.loads((ROOT / "fixtures" / "sessions_sample.json").read_text())
        items = validate_sessions(data)
        self.assertGreaterEqual(len(items), 1)
        self.assertIn("id", items[0])
        self.assertIn("title", items[0])
        self.assertIn("order", items[0])
        self.assertIn("status", items[0])

    def test_assignments_fixture(self) -> None:
        data = json.loads((ROOT / "fixtures" / "assignments_sample.json").read_text())
        items = validate_assignments(data)
        self.assertGreaterEqual(len(items), 1)
        self.assertIn("id", items[0])
        self.assertIn("title", items[0])
        self.assertIn("courseId", items[0])

    def test_empty_sessions_fail(self) -> None:
        with self.assertRaises(SchemaError):
            validate_sessions([])

    def test_session_probe_shapes(self) -> None:
        self.assertFalse(session_looks_authenticated({}))
        self.assertTrue(
            session_looks_authenticated({"user": {"email": "a@b.c"}, "expires": "x"})
        )


if __name__ == "__main__":
    unittest.main()
