"""
Tests for the CrewAI patch. CrewAI is not installed in our test environment,
so we build a tiny fake `crewai` module on the fly that exposes `Crew`,
`Agent` and `Task` classes with the methods we patch, then run the real
`patch_crewai()` against it.
"""

from __future__ import annotations
import sys
import types
import importlib
import pytest

from peekr.exporters import _exporters


class CollectingExporter:
    def __init__(self):
        self.spans = []

    def export(self, span):
        self.spans.append(span)


@pytest.fixture(autouse=True)
def isolated_exporters():
    _exporters.clear()
    col = CollectingExporter()
    _exporters.append(col)
    yield col
    _exporters.clear()


@pytest.fixture
def fake_crewai():
    """Install a stand-in `crewai` module with Crew / Agent / Task classes."""
    mod = types.ModuleType("crewai")

    class Task:
        def __init__(self, description="task", agent=None):
            self.description = description
            self.agent = agent

        def execute_sync(self, *args, **kwargs):
            agent = self.agent
            if agent is not None:
                return agent.execute_task(self)
            return "no-agent"

    class Agent:
        def __init__(self, role="researcher"):
            self.role = role

        def execute_task(self, task, *args, **kwargs):
            return f"{self.role}:{getattr(task, 'description', task)}"

    class Crew:
        def __init__(self, agents=None, tasks=None, process="sequential"):
            self.agents = list(agents or [])
            self.tasks = list(tasks or [])
            self.process = process

        def kickoff(self, inputs=None, *args, **kwargs):
            results = [t.execute_sync() for t in self.tasks]
            return " | ".join(results)

    mod.Crew = Crew
    mod.Agent = Agent
    mod.Task = Task
    sys.modules["crewai"] = mod

    # Force a fresh import of the patch module so its `_PATCHED`-style state
    # doesn't bleed across tests.
    if "peekr.patches.crewai_patch" in sys.modules:
        importlib.reload(sys.modules["peekr.patches.crewai_patch"])

    yield mod

    sys.modules.pop("crewai", None)
    if "peekr.patches.crewai_patch" in sys.modules:
        importlib.reload(sys.modules["peekr.patches.crewai_patch"])


def test_no_crewai_installed_is_noop(isolated_exporters):
    """patch_crewai() must not raise when crewai is missing."""
    sys.modules.pop("crewai", None)
    from peekr.patches.crewai_patch import patch_crewai

    patch_crewai()  # should silently return
    assert isolated_exporters.spans == []


def test_kickoff_emits_span(fake_crewai, isolated_exporters):
    from peekr.patches.crewai_patch import patch_crewai

    patch_crewai()

    agent = fake_crewai.Agent(role="planner")
    task = fake_crewai.Task(description="plan trip", agent=agent)
    crew = fake_crewai.Crew(agents=[agent], tasks=[task])
    out = crew.kickoff(inputs={"city": "NYC"})

    assert out == "planner:plan trip"
    by_name = {s.name: s for s in isolated_exporters.spans}
    kickoff = by_name["crewai.crew.kickoff"]
    assert kickoff.status == "ok"
    assert kickoff.attributes["agent_count"] == 1
    assert kickoff.attributes["task_count"] == 1
    assert "NYC" in kickoff.attributes["input"]


def test_full_span_tree(fake_crewai, isolated_exporters):
    """A kickoff fans out into task.execute → agent.execute_task; spans nest."""
    from peekr.patches.crewai_patch import patch_crewai

    patch_crewai()

    agent = fake_crewai.Agent(role="researcher")
    task = fake_crewai.Task(description="find papers", agent=agent)
    crew = fake_crewai.Crew(agents=[agent], tasks=[task])
    crew.kickoff()

    spans = isolated_exporters.spans
    names = [s.name for s in spans]
    assert "crewai.crew.kickoff" in names
    assert "crewai.task.execute" in names
    assert "crewai.agent.execute_task" in names

    by_name = {s.name: s for s in spans}
    kickoff = by_name["crewai.crew.kickoff"]
    task_span = by_name["crewai.task.execute"]
    agent_span = by_name["crewai.agent.execute_task"]

    assert task_span.parent_id == kickoff.span_id
    assert agent_span.parent_id == task_span.span_id
    assert agent_span.attributes["agent"] == "researcher"
    assert agent_span.attributes["task"] == "find papers"


def test_agent_error_marks_span(fake_crewai, isolated_exporters):
    from peekr.patches.crewai_patch import patch_crewai

    patch_crewai()

    agent = fake_crewai.Agent(role="x")

    def broken(self, task, *a, **k):
        raise RuntimeError("LLM down")

    type(agent).execute_task = broken  # patched method already applied to class

    # Re-apply since we replaced execute_task at the class level.
    patch_crewai()

    task = fake_crewai.Task(description="t", agent=agent)
    with pytest.raises(RuntimeError):
        agent.execute_task(task)

    err_spans = [
        s for s in isolated_exporters.spans if s.name == "crewai.agent.execute_task"
    ]
    assert err_spans, "expected an agent.execute_task span"
    assert err_spans[0].status == "error"
    assert "LLM down" in err_spans[0].attributes["error"]


def test_patch_is_idempotent(fake_crewai, isolated_exporters):
    """Calling patch_crewai twice must not double-wrap (one kickoff = one span)."""
    from peekr.patches.crewai_patch import patch_crewai

    patch_crewai()
    patch_crewai()

    crew = fake_crewai.Crew(agents=[], tasks=[])
    crew.kickoff()

    kickoff_spans = [
        s for s in isolated_exporters.spans if s.name == "crewai.crew.kickoff"
    ]
    assert len(kickoff_spans) == 1


def test_patched_methods_marked(fake_crewai):
    from peekr.patches.crewai_patch import patch_crewai

    patch_crewai()
    assert getattr(fake_crewai.Crew.kickoff, "_peekr_patched", False)
    assert getattr(fake_crewai.Agent.execute_task, "_peekr_patched", False)
    assert getattr(fake_crewai.Task.execute_sync, "_peekr_patched", False)
