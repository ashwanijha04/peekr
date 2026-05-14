"""Local web dashboard for peekr traces.

Launch with: ``peekr serve [--port 8000] [--db traces.db] [--jsonl traces.jsonl]``

Local-first by design: no signup, no auth, no remote backend. Reads existing
storage (SQLite by default, JSONL as fallback) and renders pages with Jinja2.
"""

from .app import create_app, run

__all__ = ["create_app", "run"]
