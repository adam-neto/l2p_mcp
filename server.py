from __future__ import annotations

from l2p import DomainBuilder, TaskBuilder
from l2p.utils.pddl_parser import parse_action
from mcp.server.fastmcp import FastMCP

import server_helpers


mcp = FastMCP(
    name="l2p",
    instructions=(
        "Provides deterministic l2p parsing, merging, and PDDL generation "
        "tools for iterative human-in-the-loop planning workflows."
    ),
)


@mcp.tool()
def parse_domain_fragment(
    domain_response: str | None = None,
    domain_response_path: str | None = None,
    use_type_hierarchy: bool = False,
) -> dict:
    """Parse domain text into structured domain components.

    This tool reads model-generated domain text and extracts whichever domain
    sections are present. It currently looks for:
    - types
    - constants
    - predicates
    - functions

    The result is returned as a fragment dictionary so the client can merge it
    into an existing domain state instead of rebuilding the whole domain from
    scratch.
    """
    text = server_helpers.load_text(
        domain_response, domain_response_path, "domain_response"
    )
    return {
        "fragment": server_helpers.parse_domain_fragment_text(
            text, use_type_hierarchy
        )
    }


@mcp.tool()
def parse_action_fragment(
    action_name: str,
    action_response: str | None = None,
    action_response_path: str | None = None,
    parameter_heading: str = "Action Parameters",
    precondition_heading: str = "Action Preconditions",
    effect_heading: str = "Action Effects",
) -> dict:
    """Parse a single action into a structured action object.

    This tool reads model-generated text for one action and extracts:
    - action parameters
    - action preconditions
    - action effects

    It returns the parsed action wrapped in an `actions` fragment so the client
    can merge it into a larger domain state later.
    """
    text = server_helpers.load_text(
        action_response, action_response_path, "action_response"
    )
    action = parse_action(
        text,
        action_name=action_name,
        param_head=parameter_heading,
        precon_head=precondition_heading,
        effect_head=effect_heading,
    )
    return {"fragment": {"actions": [action]}}


@mcp.tool()
def parse_task_fragment(
    task_response: str | None = None,
    task_response_path: str | None = None,
) -> dict:
    """Parse task text into structured task components.

    This tool reads model-generated problem or task text and extracts whichever
    task sections are present. It currently looks for:
    - objects
    - initial state
    - goal state

    The result is returned as a fragment dictionary so the client can update
    only part of a task, such as replacing the goal or adding new objects.
    """
    text = server_helpers.load_text(task_response, task_response_path, "task_response")
    return {"fragment": server_helpers.parse_task_fragment_text(text)}


@mcp.tool()
def merge_domain(
    domain: dict | None = None,
    fragment: dict | None = None,
    replace_fields: list[str] | None = None,
) -> dict:
    """Merge a parsed fragment into the current structured domain state.

    This tool combines a new parsed fragment with the current domain structure.
    It updates simple fields like types and constants, and upserts named entries
    like predicates, functions, and actions. If `replace_fields` is provided,
    those fields are replaced entirely instead of being merged.
    """
    if fragment is None:
        raise ValueError("`fragment` is required.")
    return {"domain": server_helpers.merge_domain_fragment(domain, fragment, replace_fields)}


@mcp.tool()
def merge_task(
    task: dict | None = None,
    fragment: dict | None = None,
    replace_fields: list[str] | None = None,
) -> dict:
    """Merge a parsed fragment into the current structured task state.

    This tool combines a new parsed fragment with the current task structure.
    It updates objects and merges or replaces initial and goal states depending
    on the `replace_fields` argument. This is useful for iterative edits such as
    changing only the goal without rebuilding the entire task.
    """
    if fragment is None:
        raise ValueError("`fragment` is required.")
    return {"task": server_helpers.merge_task_fragment(task, fragment, replace_fields)}


@mcp.tool()
def generate_requirements(
    domain: dict | None = None,
    types: dict | list[dict] | None = None,
    functions: list[dict] | None = None,
    actions: list[dict] | None = None,
) -> dict:
    """Infer the PDDL `:requirements` list from the current domain state.

    This tool uses `l2p`'s deterministic requirement generation logic to infer
    which PDDL requirements are needed based on the structured domain data,
    especially the types, functions, and actions currently present.
    """
    fields = server_helpers.coalesce_fields(
        domain,
        types=types,
        functions=functions,
        actions=actions,
    )
    requirements = DomainBuilder().generate_requirements(
        types=fields.get("types"),
        functions=fields.get("functions"),
        actions=fields.get("actions"),
    )
    return {"requirements": requirements}


@mcp.tool()
def generate_domain(
    domain_name: str,
    domain: dict | None = None,
    types: dict | list[dict] | None = None,
    constants: dict | None = None,
    predicates: list[dict] | None = None,
    functions: list[dict] | None = None,
    actions: list[dict] | None = None,
    requirements: list[str] | None = None,
) -> dict:
    """Generate a full PDDL domain string from structured domain data.

    This tool takes a structured domain state, or individual domain fields, and
    renders them into a complete PDDL domain definition. It is typically used
    after parsing and merging fragments into the current domain state.
    """
    fields = server_helpers.coalesce_fields(
        domain,
        types=types,
        constants=constants,
        predicates=predicates,
        functions=functions,
        actions=actions,
        requirements=requirements,
    )
    domain_pddl = DomainBuilder().generate_domain(
        domain_name=domain_name,
        types=fields.get("types"),
        constants=fields.get("constants"),
        predicates=fields.get("predicates"),
        functions=fields.get("functions"),
        actions=fields.get("actions", []),
        requirements=fields.get("requirements"),
    )
    return {"domain": domain_pddl}


@mcp.tool()
def generate_task(
    domain_name: str,
    problem_name: str,
    task: dict | None = None,
    objects: dict[str, str] | None = None,
    initial: list[str] | list[dict] | None = None,
    goal: list[str] | list[dict] | None = None,
    metric: str | None = None,
) -> dict:
    """Generate a full PDDL problem string from structured task data.

    This tool takes a structured task state, or individual task fields, and
    renders them into a complete PDDL problem definition. It expects objects,
    initial state, and goal state to be present, and is typically used after
    parsing and merging task fragments.
    """
    if metric is not None:
        raise ValueError("`metric` is not supported by l2p.TaskBuilder.generate_task.")

    fields = server_helpers.coalesce_fields(
        task, objects=objects, initial=initial, goal=goal
    )
    if fields.get("objects") is None or not isinstance(fields["objects"], dict):
        raise ValueError("`objects` is required and must be a dict[str, str].")
    if fields.get("initial") is None or not all(
        isinstance(item, dict) for item in fields["initial"]
    ):
        raise ValueError("`initial` is required and must be a list of dicts.")
    if fields.get("goal") is None or not all(isinstance(item, dict) for item in fields["goal"]):
        raise ValueError("`goal` is required and must be a list of dicts.")

    task_pddl = TaskBuilder().generate_task(
        domain_name=domain_name,
        problem_name=problem_name,
        objects=fields["objects"],
        initial=server_helpers.normalize_states(fields["initial"]),
        goal=server_helpers.normalize_states(fields["goal"]),
    )
    return {"task": task_pddl}


if __name__ == "__main__":
    mcp.run()
