import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from tests.test_offline import (
    ACTION_FRAGMENT,
    DOMAIN_FRAGMENT,
    GOAL_ONLY_FRAGMENT,
    MULTI_ACTION_FRAGMENT,
    TASK_FRAGMENT,
)


TYPE_HIERARCHY_FRAGMENT = """### TYPES
```python
[
    {
        "name": "entity",
        "children": [
            {
                "name": "vehicle",
                "children": [{"name": "robot", "children": []}]
            },
            {
                "name": "place",
                "children": [{"name": "location", "children": []}]
            }
        ]
    },
    {
        "name": "item",
        "children": [{"name": "package", "children": []}]
    }
]
```"""


# Turns an MCP tool response into plain JSON text for assertions
def tool_payload(result) -> str:
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent)
    parts: list[str] = []
    for item in result.content or []:
        parts.append(getattr(item, "text", str(item)))
    return "\n".join(parts)


class McpOfflineTests(unittest.TestCase):
    # Opens a temporary stdio MCP client connected to the local server
    def _open_session(self):
        env = dict(os.environ)
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = "." if not pythonpath else f".:{pythonpath}"

        server_params = StdioServerParameters(
            command="python3.10",
            args=["server/server.py"],
            env=env,
            cwd=Path.cwd(),
        )

        return stdio_client(server_params)

    # Calls one MCP tool and returns its decoded JSON result
    async def _call_tool(self, session: ClientSession, name: str, args: dict) -> dict:
        result = await session.call_tool(name, args)
        self.assertFalse(result.isError, tool_payload(result))
        return json.loads(tool_payload(result))

    # Runs one async test body inside a fully initialized MCP session
    async def _run_with_session(self, callback) -> None:
        async with self._open_session() as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                await callback(session)

    # Checks that the MCP server advertises the expected high level tools
    def test_lists_the_two_high_level_tools(self) -> None:
        """Show that the MCP server exposes only the high-level create-or-update entry points."""
        anyio.run(self._test_lists_the_two_high_level_tools)

    # Lists tools through the temporary MCP client
    async def _test_lists_the_two_high_level_tools(self) -> None:
        async def exercise(session: ClientSession) -> None:
            tools = await session.list_tools()
            tool_names = sorted(tool.name for tool in tools.tools)
            self.assertEqual(tool_names, ["update_domain", "update_task"])

        await self._run_with_session(exercise)

    # Checks that the MCP client can create a domain and get final domain PDDL back
    def test_creates_a_domain_from_scratch_and_generates_pddl(self) -> None:
        """Send one domain update through MCP and verify parsing, requirement inference, and final domain PDDL generation."""
        anyio.run(self._test_creates_a_domain_from_scratch_and_generates_pddl)

    # Exercises domain creation through the MCP client
    async def _test_creates_a_domain_from_scratch_and_generates_pddl(self) -> None:
        async def exercise(session: ClientSession) -> None:
            updated_domain = await self._call_tool(
                session,
                "update_domain",
                {
                    "domain_update": f"{DOMAIN_FRAGMENT}\n\n{ACTION_FRAGMENT}",
                    "action_name": "move",
                    "domain_name": "robot",
                },
            )
            self.assertIn("types", updated_domain["fragment"])
            self.assertEqual(updated_domain["fragment"]["actions"][0]["name"], "move")
            self.assertIn(":typing", updated_domain["requirements"])
            self.assertIn(":numeric-fluents", updated_domain["requirements"])
            self.assertIn("(define (domain robot)", updated_domain["domain_pddl"])
            self.assertIn("(:action move", updated_domain["domain_pddl"])

        await self._run_with_session(exercise)

    # Checks that type hierarchy updates can be parsed through the MCP client
    def test_accepts_type_hierarchy_domain_updates_through_mcp(self) -> None:
        """Parse a type hierarchy through MCP so clients can build domains from hierarchical type trees when needed."""
        anyio.run(self._test_accepts_type_hierarchy_domain_updates_through_mcp)

    # Exercises hierarchical type parsing through the MCP client
    async def _test_accepts_type_hierarchy_domain_updates_through_mcp(self) -> None:
        async def exercise(session: ClientSession) -> None:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
                handle.write(TYPE_HIERARCHY_FRAGMENT)
                hierarchy_path = handle.name

            try:
                hierarchy_update = await self._call_tool(
                    session,
                    "update_domain",
                    {
                        "domain_update_path": hierarchy_path,
                        "use_type_hierarchy": True,
                        "infer_requirements": False,
                    },
                )
                self.assertEqual(hierarchy_update["fragment"]["types"][0]["name"], "entity")
                self.assertEqual(
                    hierarchy_update["fragment"]["types"][0]["children"][0]["name"],
                    "vehicle",
                )
                self.assertNotIn("requirements", hierarchy_update)
            finally:
                Path(hierarchy_path).unlink(missing_ok=True)

        await self._run_with_session(exercise)

    # Checks that an existing domain can be patched from a file based action update
    def test_updates_an_existing_domain_with_path_input_and_action_upsert(self) -> None:
        """Create a base domain, then patch it through file-based MCP input while exercising action upserts and final domain generation."""
        anyio.run(self._test_updates_an_existing_domain_with_path_input_and_action_upsert)

    # Exercises action upserts through the MCP client
    async def _test_updates_an_existing_domain_with_path_input_and_action_upsert(self) -> None:
        async def exercise(session: ClientSession) -> None:
            base = await self._call_tool(
                session,
                "update_domain",
                {
                    "domain_update": DOMAIN_FRAGMENT,
                    "infer_requirements": False,
                },
            )
            self.assertNotIn("requirements", base)

            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
                handle.write(ACTION_FRAGMENT)
                action_path = handle.name

            try:
                action_update = await self._call_tool(
                    session,
                    "update_domain",
                    {
                        "domain_update_path": action_path,
                        "domain": base["domain"],
                        "action_name": "move",
                        "domain_name": "robot",
                    },
                )
                self.assertEqual(action_update["domain"]["actions"][0]["name"], "move")
                self.assertIn("(define (domain robot)", action_update["domain_pddl"])
            finally:
                Path(action_path).unlink(missing_ok=True)

        await self._run_with_session(exercise)

    # Checks that multiple action blocks can be parsed and merged through the MCP client
    def test_updates_a_domain_with_multiple_actions_through_mcp(self) -> None:
        """Send multiple action updates in one MCP call and verify that both actions are parsed and generated."""
        anyio.run(self._test_updates_a_domain_with_multiple_actions_through_mcp)

    # Exercises multi action parsing through the MCP client
    async def _test_updates_a_domain_with_multiple_actions_through_mcp(self) -> None:
        async def exercise(session: ClientSession) -> None:
            base = await self._call_tool(
                session,
                "update_domain",
                {
                    "domain_update": """### New Predicates
```text
(package-at ?p - package ?l - location): package is at a location
(carrying ?r - robot ?p - package): robot is carrying a package
```""",
                    "infer_requirements": False,
                },
            )

            updated = await self._call_tool(
                session,
                "update_domain",
                {
                    "domain_update": MULTI_ACTION_FRAGMENT,
                    "domain": base["domain"],
                    "domain_name": "robot",
                },
            )
            self.assertEqual(len(updated["domain"]["actions"]), 2)
            self.assertEqual(updated["domain"]["actions"][0]["name"], "move")
            self.assertEqual(updated["domain"]["actions"][1]["name"], "pick-up")
            self.assertIn("(:action move", updated["domain_pddl"])
            self.assertIn("(:action pick-up", updated["domain_pddl"])

        await self._run_with_session(exercise)

    # Checks that the MCP client can create a task and get final task PDDL back
    def test_creates_a_task_from_scratch_and_generates_problem_pddl(self) -> None:
        """Send a complete task update through MCP and verify that the parsed objects, initial state, goal, and problem PDDL come back."""
        anyio.run(self._test_creates_a_task_from_scratch_and_generates_problem_pddl)

    # Exercises task creation through the MCP client
    async def _test_creates_a_task_from_scratch_and_generates_problem_pddl(self) -> None:
        async def exercise(session: ClientSession) -> None:
            updated_task = await self._call_tool(
                session,
                "update_task",
                {
                    "task_update": TASK_FRAGMENT,
                    "domain_name": "robot",
                    "problem_name": "deliver",
                },
            )
            self.assertEqual(updated_task["task"]["objects"]["r1"], "robot")
            self.assertEqual(updated_task["task"]["initial"][0]["pred_name"], "at")
            self.assertEqual(updated_task["task"]["goal"][0]["pred_name"], "delivered")
            self.assertIn("(problem deliver)", updated_task["task_pddl"])

        await self._run_with_session(exercise)

    # Checks that an existing task can be patched from a file based goal update
    def test_updates_an_existing_task_with_path_input_and_goal_replacement(self) -> None:
        """Create a base task, then patch only the goal through file-based MCP input and verify that replacement semantics are preserved."""
        anyio.run(self._test_updates_an_existing_task_with_path_input_and_goal_replacement)

    # Exercises goal replacement through the MCP client
    async def _test_updates_an_existing_task_with_path_input_and_goal_replacement(self) -> None:
        async def exercise(session: ClientSession) -> None:
            base = await self._call_tool(
                session,
                "update_task",
                {"task_update": TASK_FRAGMENT},
            )

            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
                handle.write(GOAL_ONLY_FRAGMENT)
                goal_path = handle.name

            try:
                updated = await self._call_tool(
                    session,
                    "update_task",
                    {
                        "task_update_path": goal_path,
                        "task": base["task"],
                        "replace_fields": ["goal"],
                        "domain_name": "robot",
                        "problem_name": "deliver",
                    },
                )
                self.assertEqual(len(updated["task"]["goal"]), 2)
                self.assertEqual(updated["task"]["goal"][0]["pred_name"], "at")
                self.assertIn("(at r1 l2)", updated["task_pddl"])
            finally:
                Path(goal_path).unlink(missing_ok=True)

        await self._run_with_session(exercise)

    # Checks that domain side MCP errors are surfaced to the client
    def test_reports_domain_errors_through_mcp(self) -> None:
        """Show the domain-side failure modes that a client will see when action naming or update content is missing."""
        anyio.run(self._test_reports_domain_errors_through_mcp)

    # Exercises domain error handling through the MCP client
    async def _test_reports_domain_errors_through_mcp(self) -> None:
        async def exercise(session: ClientSession) -> None:
            action_name_error = await session.call_tool(
                "update_domain",
                {"domain_update": ACTION_FRAGMENT},
            )
            self.assertTrue(action_name_error.isError)
            self.assertIn("action_name", tool_payload(action_name_error))

            empty_update_error = await session.call_tool(
                "update_domain",
                {"domain_update": "no parseable domain content here"},
            )
            self.assertTrue(empty_update_error.isError)
            self.assertIn("No recognizable domain update sections", tool_payload(empty_update_error))

        await self._run_with_session(exercise)

    # Checks that task side MCP errors are surfaced to the client
    def test_reports_task_errors_through_mcp(self) -> None:
        """Show the task-side failure modes that a client will see for unsupported metrics, empty updates, and incomplete generation state."""
        anyio.run(self._test_reports_task_errors_through_mcp)

    # Exercises task error handling through the MCP client
    async def _test_reports_task_errors_through_mcp(self) -> None:
        async def exercise(session: ClientSession) -> None:
            metric_error = await session.call_tool(
                "update_task",
                {
                    "task_update": TASK_FRAGMENT,
                    "metric": "minimize total-cost",
                },
            )
            self.assertTrue(metric_error.isError)
            self.assertIn("metric", tool_payload(metric_error))

            empty_update_error = await session.call_tool(
                "update_task",
                {"task_update": "no parseable task content here"},
            )
            self.assertTrue(empty_update_error.isError)
            self.assertIn("No recognizable task update sections", tool_payload(empty_update_error))

            missing_state_error = await session.call_tool(
                "update_task",
                {
                    "task_update": GOAL_ONLY_FRAGMENT,
                    "domain_name": "robot",
                    "problem_name": "deliver",
                },
            )
            self.assertTrue(missing_state_error.isError)
            self.assertIn("objects", tool_payload(missing_state_error))

        await self._run_with_session(exercise)


# Runs the MCP integration tests with verbose output
if __name__ == "__main__":
    unittest.main(verbosity=2)
