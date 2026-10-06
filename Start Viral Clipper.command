#!/bin/bash
# Double-click this file to start Viral Clipper. Close the window to stop it.
cd "$(dirname "$0")"
(sleep 2 && open "http://localhost:8765") &
exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8765
