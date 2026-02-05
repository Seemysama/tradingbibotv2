"""Dashboard module - Real-time trading visualization."""
from src.dashboard.server import app, state, manager, run_dashboard

__all__ = ["app", "state", "manager", "run_dashboard"]
