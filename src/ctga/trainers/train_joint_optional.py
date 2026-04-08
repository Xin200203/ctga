"""Optional joint fine-tuning entry point."""

from __future__ import annotations


def train_joint_step(*args, **kwargs):
    raise NotImplementedError("Joint fine-tuning is intentionally deferred until stage gates are met.")
