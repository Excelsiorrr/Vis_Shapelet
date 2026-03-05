"""Backward-compatible import surface for the refactored Part A backend."""

from backend.api.part_a import router, service
from backend.app import app, create_app

__all__ = ["app", "create_app", "router", "service"]
