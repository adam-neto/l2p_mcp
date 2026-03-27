import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import server


DOMAIN_FRAGMENT = """### TYPES
```python
{"robot": "mobile agent", "location": "place", "package": "item"}
```

### CONSTANTS
```python
{"depot": "location"}
```

### New Predicates
```text
(at ?r - robot ?l - location): robot is at a location
(carrying ?r - robot ?p - package): robot carries a package
(delivered ?p - package ?l - location): package is delivered
```

### FUNCTIONS
```text
(battery-level ?r - robot) - number: battery level of robot
```"""

ACTION_FRAGMENT = """### Action Parameters
```text
?r - robot: robot
?from - location: source
?to - location: destination
```

### Action Preconditions
```lisp
(and
  (at ?r ?from)
)
```

### Action Effects
```lisp
(and
  (not (at ?r ?from))
  (at ?r ?to)
)
```"""

MULTI_ACTION_FRAGMENT = """move
### Action Parameters
```text
?r - robot: robot
?from - location: source
?to - location: destination
```

### Action Preconditions
```lisp
(and
  (at ?r ?from)
)
```

### Action Effects
```lisp
(and
  (not (at ?r ?from))
  (at ?r ?to)
)
```

## NEXT ACTION
pick-up
### Action Parameters
```text
?r - robot: robot
?p - package: package
?l - location: location
```

### Action Preconditions
```lisp
(and
  (at ?r ?l)
  (package-at ?p ?l)
)
```

### Action Effects
```lisp
(and
  (carrying ?r ?p)
)
```"""

EMPTY_ACTION_FRAGMENT = """### Action Parameters
```text
```

### Action Preconditions
```lisp
```

### Action Effects
```lisp
```
"""

TASK_FRAGMENT = """### OBJECTS
```text
r1 - robot
l1 - location
l2 - location
p1 - package
```

### INITIAL
```lisp
(at r1 l1)
(= (battery-level r1) 100)
```

### GOAL
```lisp
(delivered p1 l2)
```"""

GOAL_ONLY_FRAGMENT = """### GOAL
```lisp
(and
  (at r1 l2)
  (delivered p1 l2)
)
```"""


class AdapterOfflineTests(unittest.TestCase):
    # Checks that one domain update can be parsed merged and turned into domain PDDL
    def test_update_domain_parses_merges_and_generates(self) -> None:
        """Create a domain from model text, infer requirements, and emit domain PDDL."""
        result = server.update_domain(
            domain_update=f"{DOMAIN_FRAGMENT}\n\n{ACTION_FRAGMENT}",
            action_name="move",
            domain_name="robot",
        )

        fragment = result["fragment"]
        domain = result["domain"]

        self.assertIn("types", fragment)
        self.assertEqual(fragment["constants"]["depot"], "location")
        self.assertEqual(fragment["actions"][0]["name"], "move")
        self.assertEqual(domain["actions"][0]["params"]["?r"], "robot")
        self.assertIn(":typing", result["requirements"])
        self.assertIn(":numeric-fluents", result["requirements"])
        self.assertIn("(define (domain robot)", result["domain_pddl"])
        self.assertIn("(:action move", result["domain_pddl"])

    # Checks that an action only patch can be added without removing existing predicates
    def test_update_domain_can_upsert_action_without_replacing_predicates(self) -> None:
        """Merge an action-only update into an existing domain without losing parsed predicates."""
        base = server.update_domain(domain_update=DOMAIN_FRAGMENT)["domain"]

        updated = server.update_domain(
            domain_update=ACTION_FRAGMENT,
            domain=base,
            action_name="move",
        )["domain"]

        self.assertEqual(len(updated["predicates"]), 3)
        self.assertEqual(updated["actions"][0]["name"], "move")

    # Checks that multiple action updates can be parsed and merged in one call
    def test_update_domain_can_parse_multiple_actions(self) -> None:
        """Create or update a domain with multiple action blocks in a single update."""
        base = server.update_domain(
            domain_update="""### New Predicates
```text
(package-at ?p - package ?l - location): package is at a location
(carrying ?r - robot ?p - package): robot is carrying a package
```"""
        )["domain"]

        updated = server.update_domain(
            domain_update=MULTI_ACTION_FRAGMENT,
            domain=base,
            domain_name="robot",
        )

        self.assertEqual(len(updated["domain"]["actions"]), 2)
        self.assertEqual(updated["domain"]["actions"][0]["name"], "move")
        self.assertEqual(updated["domain"]["actions"][1]["name"], "pick-up")
        self.assertIn("(:action move", updated["domain_pddl"])
        self.assertIn("(:action pick-up", updated["domain_pddl"])

    # Checks that action updates fail when the action name is missing
    def test_update_domain_requires_action_name_for_action_sections(self) -> None:
        """Reject action-shaped domain updates when the caller does not name the action to patch."""
        with self.assertRaisesRegex(ValueError, "action_name"):
            server.update_domain(domain_update=ACTION_FRAGMENT)

    # Checks that multiple unnamed action blocks can use a matching list of action names
    def test_update_domain_can_use_multiple_action_names(self) -> None:
        """Parse multiple unnamed action blocks by passing a matching list of action names."""
        unnamed_actions = MULTI_ACTION_FRAGMENT.replace("move\n", "", 1).replace(
            "\n## NEXT ACTION\npick-up", "\n## NEXT ACTION", 1
        )

        updated = server.update_domain(
            domain_update=unnamed_actions,
            action_name=["move", "pick-up"],
        )

        self.assertEqual(updated["fragment"]["actions"][0]["name"], "move")
        self.assertEqual(updated["fragment"]["actions"][1]["name"], "pick-up")

    # Checks that action sections can be present but empty when there are zero updates for that feature
    def test_update_domain_can_parse_empty_action_sections(self) -> None:
        """Allow action parameters preconditions and effects to be empty when there are zero updates."""
        updated = server.update_domain(
            domain_update=EMPTY_ACTION_FRAGMENT,
            action_name="noop",
        )

        self.assertEqual(updated["fragment"]["actions"][0]["name"], "noop")
        self.assertEqual(updated["fragment"]["actions"][0]["params"], {})
        self.assertEqual(updated["fragment"]["actions"][0]["preconditions"], "")
        self.assertEqual(updated["fragment"]["actions"][0]["effects"], "")

    # Checks that one task update can be parsed merged and turned into task PDDL
    def test_update_task_parses_merges_and_generates(self) -> None:
        """Create a task from model text and emit problem PDDL once objects, initial state, and goal exist."""
        result = server.update_task(
            task_update=TASK_FRAGMENT,
            domain_name="robot",
            problem_name="deliver",
        )

        fragment = result["fragment"]
        task = result["task"]

        self.assertEqual(fragment["objects"]["r1"], "robot")
        self.assertEqual(task["initial"][0]["pred_name"], "at")
        self.assertEqual(task["initial"][1]["func_name"], "battery-level")
        self.assertEqual(task["goal"][0]["pred_name"], "delivered")
        self.assertIn("(problem deliver)", result["task_pddl"])
        self.assertIn("(delivered p1 l2)", result["task_pddl"])

    # Checks that replacing only the goal leaves the rest of the task unchanged
    def test_update_task_can_replace_goal(self) -> None:
        """Replace an existing goal fragment while preserving the rest of the task state."""
        base = server.update_task(task_update=TASK_FRAGMENT)["task"]

        updated = server.update_task(
            task_update=GOAL_ONLY_FRAGMENT,
            task=base,
            replace_fields=["goal"],
            domain_name="robot",
            problem_name="deliver",
        )

        self.assertEqual(len(updated["task"]["goal"]), 2)
        self.assertEqual(updated["task"]["goal"][0]["pred_name"], "at")
        self.assertIn("(at r1 l2)", updated["task_pddl"])

    # Checks that unsupported task metrics still raise a clear error
    def test_update_task_rejects_metric(self) -> None:
        """Surface the current TaskBuilder limitation that metric generation is not supported."""
        with self.assertRaisesRegex(ValueError, "metric"):
            server.update_task(
                task_update=TASK_FRAGMENT,
                metric="minimize total-cost",
            )

    # Checks that task generation fails until the full task state is present
    def test_update_task_requires_full_state_for_generation(self) -> None:
        """Refuse to generate task PDDL when a partial update does not yet include the full task state."""
        with self.assertRaisesRegex(ValueError, "objects"):
            server.update_task(
                task_update=GOAL_ONLY_FRAGMENT,
                domain_name="robot",
                problem_name="deliver",
            )


# Runs the offline unit tests with verbose output
if __name__ == "__main__":
    unittest.main(verbosity=2)
