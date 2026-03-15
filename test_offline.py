from __future__ import annotations

import os
import unittest
from unittest.mock import patch

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

    def test_build_llm_uses_unified_llm_defaults(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
            with patch("l2p.llm.unified.UnifiedLLM") as unified_llm:
                sentinel = object()
                unified_llm.return_value = sentinel

                result = server._build_llm(
                    provider="openai",
                    model="gpt-4o-mini",
                    api_key=None,
                    api_key_env="OPENAI_API_KEY",
                    config_path=None,
                )

        self.assertIs(result, sentinel)
        kwargs = unified_llm.call_args.kwargs
        self.assertEqual(kwargs["provider"], "openai")
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        self.assertTrue(kwargs["config_path"].endswith("l2p/llm/utils/llm.yaml"))
        self.assertEqual(kwargs["api_key"], "env-key")

    def test_build_llm_passes_api_key_from_env(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}, clear=True):
            with patch("l2p.llm.unified.UnifiedLLM") as unified_llm:
                server._build_llm(
                    provider="anthropic",
                    model="claude-3-5-sonnet",
                    api_key=None,
                    api_key_env="ANTHROPIC_API_KEY",
                    config_path="/tmp/custom-llm.yaml",
                )

        self.assertEqual(
            unified_llm.call_args.kwargs,
            {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet",
                "config_path": "/tmp/custom-llm.yaml",
                "api_key": "env-key",
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
