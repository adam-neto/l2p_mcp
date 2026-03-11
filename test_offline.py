from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from l2p.llm.base import BaseLLM
import server


class FakeModel(BaseLLM):
    provider = "fake"

    def __init__(self, output: str):
        self.output = output
        super().__init__("fake-model", api_key="fake-key")

    def valid_models(self) -> list[str]:
        return ["fake-model"]

    def query(self, prompt: str) -> str:
        return self.output

    def reset_tokens(self) -> None:
        pass


PREDICATE_OUTPUT = """### New Predicates
```
(at ?r - robot ?l - location): robot is at location
(connected ?from - location ?to - location): locations are connected
```"""

TASK_OUTPUT = """### OBJECTS
```
r1 - robot
l1 - location
l2 - location
```

### INITIAL
```
(at r1 l1)
(connected l1 l2)
```

### GOAL
```
(at r1 l2)
```"""

FEEDBACK_OUTPUT = "No feedback needed."

SERVER_BOOTSTRAP = f"""
from l2p.llm.base import BaseLLM
import server

RESPONSES = {json.dumps([PREDICATE_OUTPUT, TASK_OUTPUT, FEEDBACK_OUTPUT])}

class FakeModel(BaseLLM):
    provider = "fake"

    def __init__(self):
        super().__init__("fake-model", api_key="fake-key")

    def valid_models(self):
        return ["fake-model"]

    def query(self, prompt):
        if not RESPONSES:
            raise RuntimeError("FakeModel ran out of prepared responses.")
        return RESPONSES.pop(0)

    def reset_tokens(self):
        pass

server._build_llm = lambda *args, **kwargs: FakeModel()

from server import main
main()
"""


def tool_payload(result) -> str:
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent)
    parts: list[str] = []
    for item in result.content or []:
        parts.append(getattr(item, "text", str(item)))
    return "\n".join(parts)


class AdapterOfflineTests(unittest.TestCase):
    def test_formalize_domain_predicates(self) -> None:
        with patch.object(
            server, "_build_llm", return_value=FakeModel(PREDICATE_OUTPUT)
        ):
            result = server.formalize_domain_predicates(
                domain_description="Robot moves between locations.",
                prompt_template="Domain: {domain_desc}",
            )

        self.assertEqual(result[0][0]["name"], "at")
        self.assertEqual(result[0][1]["name"], "connected")
        self.assertEqual(result[2], [True, "All validations passed."])

    def test_formalize_task(self) -> None:
        with patch.object(server, "_build_llm", return_value=FakeModel(TASK_OUTPUT)):
            result = server.formalize_task(
                problem_description="Move r1 from l1 to l2.",
                prompt_template="Problem: {problem_desc}",
            )

        self.assertEqual(result[0]["r1"], "robot")
        self.assertEqual(result[1][0]["pred_name"], "at")
        self.assertEqual(result[4], [True, "All validations passed."])

    def test_generate_task(self) -> None:
        result = server.generate_task(
            domain_name="robot",
            problem_name="move-r1",
            objects={"r1": "robot", "l1": "location", "l2": "location"},
            initial=[
                {"name": "at", "params": ["r1", "l1"], "neg": False},
                {"name": "connected", "params": ["l1", "l2"], "neg": False},
            ],
            goal=[{"name": "at", "params": ["r1", "l2"], "neg": False}],
        )

        self.assertIn("(problem move-r1)", result["task"])
        self.assertIn("(at r1 l2)", result["task"])

    def test_task_feedback(self) -> None:
        with patch.object(
            server, "_build_llm", return_value=FakeModel(FEEDBACK_OUTPUT)
        ):
            result = server.task_feedback(
                problem_description="Move r1 from l1 to l2.",
                llm_output=TASK_OUTPUT,
                feedback_template="Review {llm_output}",
            )

        self.assertEqual(result, [True, FEEDBACK_OUTPUT])

    def test_run_fast_downward_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            planner = tmp / "fake-downward"
            domain = tmp / "domain.pddl"
            problem = tmp / "problem.pddl"
            planner.write_text("#!/bin/sh\necho 'move (r1 l1 l2)'\n", encoding="utf-8")
            planner.chmod(0o755)
            domain.write_text("(define (domain robot))\n", encoding="utf-8")
            problem.write_text("(define (problem move-r1))\n", encoding="utf-8")

            result = server.run_fast_downward(
                domain_file=str(domain),
                problem_file=str(problem),
                planner_path=str(planner),
            )

        self.assertEqual(result["plan"][0], True)
        self.assertIn("move (r1 l1 l2)", result["plan"][1])

    def test_generate_task_rejects_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "metric"):
            server.generate_task(
                domain_name="robot",
                problem_name="move-r1",
                objects={"r1": "robot"},
                initial=[{"name": "at", "params": ["r1", "l1"], "neg": False}],
                goal=[{"name": "at", "params": ["r1", "l2"], "neg": False}],
                metric="minimize total-cost",
            )

    def test_generate_task_rejects_bad_objects(self) -> None:
        with self.assertRaisesRegex(ValueError, "objects"):
            server.generate_task(
                domain_name="robot",
                problem_name="move-r1",
                objects=None,
                initial=[{"name": "at", "params": ["r1", "l1"], "neg": False}],
                goal=[{"name": "at", "params": ["r1", "l2"], "neg": False}],
            )

    def test_formalize_task_requires_prompt_or_path(self) -> None:
        with patch.object(server, "_build_llm", return_value=FakeModel(TASK_OUTPUT)):
            with self.assertRaisesRegex(ValueError, "prompt_template"):
                server.formalize_task(problem_description="Move r1.")

    def test_build_llm_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                server._build_llm(
                    model="gpt-4o-mini",
                    api_key=None,
                    api_key_env="OPENAI_API_KEY",
                    base_url=None,
                    temperature=None,
                )

    def test_run_fast_downward_missing_binary_returns_failure_tuple(self) -> None:
        result = server.run_fast_downward(
            domain_file="missing-domain.pddl",
            problem_file="missing-problem.pddl",
            planner_path="/definitely/missing/fast-downward",
        )
        self.assertEqual(result["plan"][0], False)


class McpOfflineTests(unittest.TestCase):
    def test_mcp_stdio_happy_path_and_error_cases(self) -> None:
        anyio.run(self._run_mcp_cases)

    async def _run_mcp_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            planner = tmp / "fake-downward"
            domain = tmp / "domain.pddl"
            problem = tmp / "problem.pddl"
            planner.write_text("#!/bin/sh\necho 'move (r1 l1 l2)'\n", encoding="utf-8")
            planner.chmod(0o755)
            domain.write_text("(define (domain robot))\n", encoding="utf-8")
            problem.write_text("(define (problem move-r1))\n", encoding="utf-8")

            env = dict(os.environ)
            pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = "." if not pythonpath else f".:{pythonpath}"

            server = StdioServerParameters(
                command="python3",
                args=["-c", SERVER_BOOTSTRAP],
                env=env,
                cwd=Path.cwd(),
            )

            async with stdio_client(server) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    tools = await session.list_tools()
                    tool_names = sorted(tool.name for tool in tools.tools)
                    self.assertEqual(
                        tool_names,
                        [
                            "formalize_domain_predicates",
                            "formalize_task",
                            "generate_task",
                            "run_fast_downward",
                            "task_feedback",
                        ],
                    )

                    domain_result = await session.call_tool(
                        "formalize_domain_predicates",
                        {
                            "domain_description": "Robot domain",
                            "prompt_template": "Domain: {domain_desc}",
                        },
                    )
                    self.assertFalse(domain_result.isError)
                    self.assertIn("connected", tool_payload(domain_result))

                    task_result = await session.call_tool(
                        "formalize_task",
                        {
                            "problem_description": "Move r1 from l1 to l2.",
                            "prompt_template": "Problem: {problem_desc}",
                        },
                    )
                    self.assertFalse(task_result.isError)
                    self.assertIn("r1", tool_payload(task_result))

                    generated_result = await session.call_tool(
                        "generate_task",
                        {
                            "domain_name": "robot",
                            "problem_name": "move-r1",
                            "objects": {
                                "r1": "robot",
                                "l1": "location",
                                "l2": "location",
                            },
                            "initial": [
                                {"name": "at", "params": ["r1", "l1"], "neg": False}
                            ],
                            "goal": [
                                {"name": "at", "params": ["r1", "l2"], "neg": False}
                            ],
                        },
                    )
                    self.assertFalse(generated_result.isError)
                    self.assertIn("(problem move-r1)", tool_payload(generated_result))

                    feedback_result = await session.call_tool(
                        "task_feedback",
                        {
                            "problem_description": "Move r1 from l1 to l2.",
                            "llm_output": "candidate task output",
                            "feedback_template": "Review {llm_output}",
                        },
                    )
                    self.assertFalse(feedback_result.isError)
                    self.assertIn("No feedback needed", tool_payload(feedback_result))

                    planner_result = await session.call_tool(
                        "run_fast_downward",
                        {
                            "domain_file": str(domain),
                            "problem_file": str(problem),
                            "planner_path": str(planner),
                        },
                    )
                    self.assertFalse(planner_result.isError)
                    self.assertIn("move (r1 l1 l2)", tool_payload(planner_result))

                    metric_error = await session.call_tool(
                        "generate_task",
                        {
                            "domain_name": "robot",
                            "problem_name": "move-r1",
                            "objects": {"r1": "robot"},
                            "initial": [
                                {"name": "at", "params": ["r1", "l1"], "neg": False}
                            ],
                            "goal": [
                                {"name": "at", "params": ["r1", "l2"], "neg": False}
                            ],
                            "metric": "minimize total-cost",
                        },
                    )
                    self.assertTrue(metric_error.isError)
                    self.assertIn("metric", tool_payload(metric_error))

                    bad_feedback = await session.call_tool(
                        "task_feedback",
                        {
                            "problem_description": "Move r1 from l1 to l2.",
                            "llm_output": "candidate task output",
                            "feedback_template": "Review {llm_output}",
                            "feedback_type": "task",
                        },
                    )
                    self.assertTrue(bad_feedback.isError)
                    self.assertIn("Invalid feedback_type", tool_payload(bad_feedback))

                    missing_prompt = await session.call_tool(
                        "formalize_task",
                        {
                            "problem_description": "Move r1 from l1 to l2.",
                        },
                    )
                    self.assertTrue(missing_prompt.isError)
                    self.assertIn("prompt_template", tool_payload(missing_prompt))


if __name__ == "__main__":
    unittest.main(verbosity=2)
