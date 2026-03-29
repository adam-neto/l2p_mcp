import os

from l2p import DomainBuilder, TaskBuilder
from mcp.server.fastmcp import FastMCP

try:
    from . import server_helpers
except ImportError:
    import server_helpers


mcp = FastMCP(
    name="l2p",
    instructions=(
        "Use `update_domain` when model output changes the planning domain and "
        "use `update_task` when model output changes the planning problem. "
        "These tools parse model-written updates, merge them into the current "
        "structured state, infer requirements when appropriate, and generate "
        "updated PDDL artifacts when names are provided."
    ),
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8002")),
    streamable_http_path=os.getenv("MCP_STREAMABLE_HTTP_PATH", "/mcp"),
)


# Creates or updates a domain by parsing model output and optionally generating domain PDDL
@mcp.tool()
def update_domain(
    domain_update: str | None = None,
    domain_update_path: str | None = None,
    domain: dict | None = None,
    domain_name: str | None = None,
    action_name: str | list[str] | None = None,
    replace_fields: list[str] | None = None,
    use_type_hierarchy: bool = False,
    infer_requirements: bool = True,
) -> dict:
    """Apply a model-generated domain update to the current domain state.

    Call this when the model has produced text that changes any part of the
    planning domain. The server will read the update text, parse the supported
    sections, merge them into the current `domain`, optionally infer
    `:requirements`, and optionally generate the full PDDL domain string.

    The prompt templates expect these exact headings and section shapes. 
    Each feature can be omitted, included once, or include multiple entries 
    inside its fenced block. For example, `### TYPES` may contain zero types, 
    one type, or many types. The same is true for `### CONSTANTS`, 
    `### New Predicates`, and `### FUNCTIONS`. For any domain section with 
    zero updates, keep the heading and leave the fenced block empty.

    Use the standard type template when you want flat types:

    ### TYPES
    ```python
    {
        "type_1": "description",
        "type_2": "description",
        "type_3": "description"
    }
    ```

    Use the type hierarchy template when `use_type_hierarchy=True` and you want
    nested subtype structure:

    ### TYPES
    ```python
    [
        {
            "name": "parent_type_1",
            "children": [
                {
                    "name": "child_type_1",
                    "children": [{"name": "child_child_type_1", "children": []}]
                }
            ]
        }
    ]
    ```

    Use the constants template for named constants:

    ### CONSTANTS
    ```python
    {
        "const_1": "type_1"
    }
    ```

    Use the predicate template below with one variable per type declaration and
    one predicate per list item:

    ### New Predicates
    ```text
    - (predicate_name_1 ?t1 - type_1 ?t2 - type_2): 'predicate_description'
    - (predicate_name_2 ?t3 - type_3 ?t4 - type_4): 'predicate_description'
    - (predicate_name_3 ?t5 - type_5): 'predicate_description'
    ```

    Use the function template the same way:

    ### FUNCTIONS
    ```text
    - (function_name_1 ?t1 - type_1 ?t2 - type_2): 'function_description'
    - (function_name_2 ?t3 - type_3 ?t4 - type_4): 'function_description'
    - (function_name_3 ?t5 - type_5): 'function_description'
    ```

    For action updates, the parameter template expects one parameter per line:

    ### Action Parameters
    ```text
    - ?t1 - type_1: 'parameter_description'
    - ?t2 - type_2: 'parameter_description'
    ```

    Action updates can also be omitted, provided once, or provided many times.
    For a single unnamed action block, pass one `action_name`. For multiple
    action blocks, either provide one action name before each block in the
    update text, or pass `action_name` as a list of names. The server supports
    the `## NEXT ACTION` separator shown below. For any action subsection with
    zero updates, keep the heading and leave the fenced block empty.

    Example multi action layout

    move
    ### Action Parameters
    ```text
    - ?t1 - type_1: 'parameter_description'
    ```

    ### Action Preconditions
    ```lisp
    (and
        (predicate_name ?t1) ; COMMENT DESCRIPTION
    )
    ```

    ### Action Effects
    ```lisp
    (and
        (predicate_name ?t1) ; COMMENT DESCRIPTION
    )
    ```

    ## NEXT ACTION
    load
    ### Action Parameters
    ```text
    - ?t1 - type_1: 'parameter_description'
    - ?t2 - type_2: 'parameter_description'
    ```

    ### Action Preconditions
    ```lisp
    (and
        (predicate_name ?t1 ?t2) ; COMMENT DESCRIPTION
    )
    ```

    ### Action Effects
    ```lisp
    (and
        (predicate_name ?t1 ?t2) ; COMMENT DESCRIPTION
    )
    ```

    The precondition template expects a PDDL block and may also include
    `### New Predicates` if the model introduced new predicates while building
    the action:

    ### Action Preconditions
    ```lisp
    (and
        (predicate_name ?t1 ?t2) ; COMMENT DESCRIPTION
    )
    ```

    ### New Predicates
    ```text
    - (predicate_name ?t1 - type_1 ?t2 - type_2): 'predicate_description'
    ```

    The effects template follows the same pattern and may also include
    `### New Predicates`:

    ### Action Effects
    ```lisp
    (and
        (predicate_name ?t1 ?t2) ; COMMENT DESCRIPTION
    )
    ```

    ### New Predicates
    ```text
    ```

    If the update includes unnamed action headings above, also provide
    `action_name`. Use `replace_fields` when a field such as `actions`,
    `predicates`, `types`, `constants`, or `functions` should be replaced
    instead of upserted or merged. If `domain_name` is provided, the tool also
    returns `domain_pddl`.
    """
    text = server_helpers.load_text(
        text=domain_update,
        path=domain_update_path,
        field_name="domain_update",
    )
    fragment = server_helpers.parse_domain_update_text(
        text=text,
        action_name=action_name,
        use_type_hierarchy=use_type_hierarchy,
    )
    merged_domain = server_helpers.merge_domain_fragment(
        domain=domain,
        fragment=fragment,
        replace_fields=replace_fields,
    )

    result = {"fragment": fragment, "domain": merged_domain}

    if infer_requirements:
        requirements = DomainBuilder().generate_requirements(
            types=merged_domain.get("types"),
            functions=merged_domain.get("functions"),
            actions=merged_domain.get("actions"),
        )
        merged_domain["requirements"] = requirements
        result["requirements"] = requirements

    if domain_name is not None:
        result["domain_pddl"] = DomainBuilder().generate_domain(
            domain_name=domain_name,
            requirements=merged_domain.get("requirements"),
            types=merged_domain.get("types"),
            constants=merged_domain.get("constants"),
            predicates=merged_domain.get("predicates"),
            functions=merged_domain.get("functions"),
            actions=merged_domain.get("actions", []),
        )

    return result


# Creates or updates a task by parsing model output and optionally generating task PDDL
@mcp.tool()
def update_task(
    task_update: str | None = None,
    task_update_path: str | None = None,
    task: dict | None = None,
    domain_name: str | None = None,
    problem_name: str | None = None,
    replace_fields: list[str] | None = None,
    metric: str | None = None,
) -> dict:
    """Apply a model-generated task update to the current problem/task state.

    Call this when the model has produced text that changes the planning
    problem. The server will parse the task update, merge it into the current
    `task`, and optionally generate a final PDDL problem file.

    The prompt templates expect these exact headings and section shapes. 
    Each feature can be omitted, included once, or include multiple entries 
    inside its fenced block. For example, `### OBJECTS` may contain zero 
    objects, one object, or many objects. The same is true for `### INITIAL` 
    and `### GOAL`. For any task section with zero updates, keep the heading 
    and leave the fenced block empty.

    Use the objects template with one object declaration per line and do not
    group objects by type:

    ### OBJECTS
    ```text
    object1 - type_1
    object2 - type_2
    object3 - type_1
    ```

    For a full task response the sections should appear in the order
    `### OBJECTS`, `### INITIAL`, then `### GOAL`

    Use the initial state template for predicates and numeric assignments:

    ### INITIAL
    ```lisp
    (<predicate_name> <object1> <object2>) ; comment for initial state predicate 1
    (<predicate_name> <object3> <object4>) ; comment for initial state predicate 2
    (<predicate_name> <object5>) ; comment for initial state predicate 3
    (= (<function_name> <object6>) <value>) ; comment for a numeric assignment
    ```

    Use the goal template for predicates or numeric goal conditions:

    ### GOAL
    ```lisp
    (<predicate_name> <object1> <object2>) ; comment for goal state predicate 1
    (<predicate_name> <object3> <object4>) ; comment for goal state predicate 2
    (<predicate_name> <object5>) ; comment for goal state predicate 3
    ```

    Numeric goals can also follow the template style:

    ### GOAL
    ```lisp
    (<operator> (<function_name> <object6>) <value>) ; comment for a numeric assignment
    ```

    Any subset of these sections may be provided when only part of the task
    should change. Use `replace_fields=["goal"]`, for example, when a new goal
    should replace the existing one rather than be merged into it. If both
    `domain_name` and `problem_name` are provided, the tool also returns
    `task_pddl`, but generation requires that the merged task contains
    `objects`, `initial`, and `goal`.
    """
    if metric is not None:
        raise ValueError("`metric` is not supported by l2p.TaskBuilder.generate_task.")

    text = server_helpers.load_text(
        text=task_update,
        path=task_update_path,
        field_name="task_update",
    )
    fragment = server_helpers.parse_task_update_text(text)
    merged_task = server_helpers.merge_task_fragment(
        task=task,
        fragment=fragment,
        replace_fields=replace_fields,
    )

    result = {"fragment": fragment, "task": merged_task}

    if domain_name is not None and problem_name is not None:
        required_fields = {
            "objects": isinstance(merged_task.get("objects"), dict),
            "initial": merged_task.get("initial") is not None,
            "goal": merged_task.get("goal") is not None,
        }
        for field, present in required_fields.items():
            if not present:
                raise ValueError(
                    f"Task PDDL generation requires `{field}` to be present."
                )

        result["task_pddl"] = TaskBuilder().generate_task(
            domain_name=domain_name,
            problem_name=problem_name,
            objects=merged_task["objects"],
            initial=server_helpers.normalize_states(merged_task["initial"]),
            goal=server_helpers.normalize_states(merged_task["goal"]),
        )

    return result


def main() -> None:
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    mcp.run(transport=transport)


# Starts the MCP server when the file is run directly
if __name__ == "__main__":
    main()
