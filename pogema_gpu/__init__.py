"""Explicit Pogema tasks and shared-policy evaluation.

Importing this namespace does not import Torch or compile a CUDA extension.
"""

__version__ = "0.0.0.dev0"

from .tasks import Task, load_tasks, save_tasks

__all__ = ["Task", "load_tasks", "save_tasks"]
