"""Unit tests for the project state machine (guide section 4)."""

from __future__ import annotations

import pytest

from backend.models import ProcessingError, ProjectStatus
from backend.orchestration.state import Project


def test_happy_path_transitions():
    p = Project("p1", "test")
    assert p.status == ProjectStatus.CREATED
    p.transition(ProjectStatus.UPLOADING)
    p.transition(ProjectStatus.PROCESSING)
    p.transition(ProjectStatus.COMPLETED)
    assert p.status == ProjectStatus.COMPLETED


def test_illegal_transition_rejected():
    p = Project("p2", "test")
    # CREATED cannot jump straight to COMPLETED.
    with pytest.raises(ValueError):
        p.transition(ProjectStatus.COMPLETED)


def test_fail_reachable_from_any_active_state():
    p = Project("p3", "test")
    p.transition(ProjectStatus.UPLOADING)
    p.transition(ProjectStatus.PROCESSING)
    p.fail(ProcessingError(title="boom", detail="it broke"))
    assert p.status == ProjectStatus.FAILED
    assert p.error is not None
    assert p.error.title == "boom"


def test_terminal_states_are_terminal():
    p = Project("p4", "test")
    p.transition(ProjectStatus.UPLOADING)
    p.transition(ProjectStatus.PROCESSING)
    p.transition(ProjectStatus.COMPLETED)
    with pytest.raises(ValueError):
        p.transition(ProjectStatus.PROCESSING)
