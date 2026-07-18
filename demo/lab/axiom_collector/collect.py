#!/usr/bin/env python3
"""Headless collector: Axiom course sessions + assignments via NextAuth cookie.

Auth material (never commit):
  AXIOM_SESSION_TOKEN   raw __Secure-next-auth.session-token value
  AXIOM_COOKIE_FILE     path to file with token or full Cookie header
  --cookie-file PATH    same as AXIOM_COOKIE_FILE
  --token TOKEN         raw token (prefer env/file)

Example:
  export AXIOM_SESSION_TOKEN='…'
  python collect.py --course-id cmq97i5kn032208o8xu5dab4q -o out/course.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from client import ApiError, AuthError, AxiomClient
from schema import (
    SchemaError,
    session_looks_authenticated,
    validate_assignments,
    validate_sessions,
)

DEFAULT_COURSE = "cmq97i5kn032208o8xu5dab4q"


def resolve_token(args: argparse.Namespace) -> str:
    if args.token:
        return args.token.strip()
    env_tok = os.environ.get("AXIOM_SESSION_TOKEN", "").strip()
    if env_tok:
        return env_tok
    cookie_file = args.cookie_file or os.environ.get("AXIOM_COOKIE_FILE", "")
    if cookie_file:
        path = Path(cookie_file).expanduser()
        if not path.is_file():
            raise AuthError(f"cookie file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise AuthError(f"cookie file empty: {path}")
        # Allow KEY=value lines
        if "\n" in text and "session-token" not in text.split("\n", 1)[0]:
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("AXIOM_SESSION_TOKEN="):
                    return line.split("=", 1)[1].strip().strip("'\"")
                if "=" not in line or line.startswith("__Secure"):
                    return line
        return text
    raise AuthError(
        "no session cookie: set AXIOM_SESSION_TOKEN, AXIOM_COOKIE_FILE, "
        "--cookie-file, or --token (see AUTH.md)"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect Axiom course sessions + assignments")
    p.add_argument("--course-id", default=DEFAULT_COURSE, help="Course id")
    p.add_argument("-o", "--output", type=Path, help="Write combined JSON here")
    p.add_argument(
        "--out-dir",
        type=Path,
        help="Write sessions.json + assignments.json + collect.json under this dir",
    )
    p.add_argument("--cookie-file", type=Path, help="File with session token or Cookie header")
    p.add_argument("--token", help="Raw session token (prefer env/file)")
    p.add_argument("--base-url", default="https://axiom.theschoolofai.in")
    p.add_argument(
        "--skip-session-probe",
        action="store_true",
        help="Do not call /api/auth/session before collect",
    )
    p.add_argument(
        "--allow-empty",
        action="store_true",
        help="Do not fail if sessions/assignments arrays are empty",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        token = resolve_token(args)
        client = AxiomClient(session_token=token, base_url=args.base_url)

        session_info = None
        if not args.skip_session_probe:
            session_info = client.get_session()
            if not session_looks_authenticated(session_info):
                raise AuthError(
                    "/api/auth/session returned empty/unauthenticated body {}; "
                    "cookie/session invalid or expired"
                )

        sessions_raw = client.get_sessions(args.course_id)
        assignments_raw = client.get_assignments(args.course_id)

        if args.allow_empty:
            sessions = sessions_raw if isinstance(sessions_raw, list) else []
            assignments = assignments_raw if isinstance(assignments_raw, list) else []
            if not isinstance(sessions_raw, list) or not isinstance(assignments_raw, list):
                raise SchemaError("expected list payloads for sessions and assignments")
        else:
            sessions = validate_sessions(sessions_raw)
            assignments = validate_assignments(assignments_raw)

        payload = {
            "collectedAt": datetime.now(timezone.utc).isoformat(),
            "baseUrl": args.base_url.rstrip("/"),
            "courseId": args.course_id,
            "session": session_info,
            "sessions": sessions,
            "assignments": assignments,
            "counts": {
                "sessions": len(sessions),
                "assignments": len(assignments),
            },
        }

        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / "collect.json").write_text(text, encoding="utf-8")
            (args.out_dir / "sessions.json").write_text(
                json.dumps(sessions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            (args.out_dir / "assignments.json").write_text(
                json.dumps(assignments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(
                f"wrote {args.out_dir}/collect.json "
                f"(sessions={len(sessions)} assignments={len(assignments)})",
                file=sys.stderr,
            )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(
                f"wrote {args.output} "
                f"(sessions={len(sessions)} assignments={len(assignments)})",
                file=sys.stderr,
            )
        if not args.output and not args.out_dir:
            sys.stdout.write(text)

        return 0
    except AuthError as e:
        print(f"auth/session/cookie error: {e}", file=sys.stderr)
        return 2
    except (SchemaError, ApiError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
