from copy import deepcopy
from l2p.utils.pddl_format import remove_comments
from l2p.utils.pddl_parser import (
    combine_blocks,
    parse_constants,
    parse_functions,
    parse_goal,
    parse_heading,
    parse_initial,
    parse_new_predicates,
    parse_objects,
    parse_action,
    parse_pddl,
    parse_task_states,
    parse_type_hierarchy,
    parse_types,
)
from pathlib import Path
import re
from typing import Any

# Reads inline text or a file and normalizes fenced code block labels
def load_text(text: str | None, path: str | None, field_name: str) -> str:
    if text is not None:
        source = text
    elif path is not None:
        source = Path(path).expanduser().read_text(encoding="utf-8")
    else:
        raise ValueError(f"`{field_name}` or `{field_name}_path` is required.")
    return re.sub(r"```[a-zA-Z0-9_-]+\n", "```\n", source)


# Converts task states into the structure expected by TaskBuilder
def normalize_states(
    states: list[str] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if states is None:
        return None

    normalized: list[dict[str, Any]] = []
    for state in states:
        if not isinstance(state, dict):
            raise ValueError("Task states must be provided as dictionaries.")
        if "pred_name" in state or "func_name" in state:
            normalized.append(state)
        elif "name" in state:
            normalized.append(
                {
                    "pred_name": state["name"],
                    "params": state.get("params", []),
                    "neg": state.get("neg", False),
                }
            )
        else:
            raise ValueError(
                "Task state dictionaries must include `name` or `pred_name`."
            )
    return normalized


# Merges lists of named entries such as predicates functions and actions
def merge_named_entries(
    current: list[dict[str, Any]] | None, incoming: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    merged = deepcopy(current or [])
    by_name: dict[str, int] = {}

    for index, item in enumerate(merged):
        name = item.get("name")
        if isinstance(name, str):
            by_name[name] = index

    for item in incoming or []:
        copied = deepcopy(item)
        name = item.get("name")
        if isinstance(name, str) and name in by_name:
            merged[by_name[name]] = copied
        else:
            merged.append(copied)
            if isinstance(name, str):
                by_name[name] = len(merged) - 1

    return merged


# Builds a stable key for deduplicating predicate and function states
def state_key(state: dict[str, Any]) -> tuple[Any, ...]:
    if "func_name" in state:
        return (
            "func",
            state.get("func_name"),
            tuple(state.get("params", [])),
            state.get("op"),
            state.get("value"),
        )
    return (
        "pred",
        state.get("pred_name") or state.get("name"),
        tuple(state.get("params", [])),
        bool(state.get("neg", False)),
    )


# Merges task state lists while replacing duplicate entries by key
def merge_states(
    current: list[dict[str, Any]] | None, incoming: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    merged = deepcopy(current or [])
    index: dict[tuple[Any, ...], int] = {}

    for i, item in enumerate(merged):
        index[state_key(item)] = i

    for item in incoming or []:
        copied = deepcopy(item)
        key = state_key(copied)
        if key in index:
            merged[index[key]] = copied
        else:
            merged.append(copied)
            index[key] = len(merged) - 1

    return merged


# Merges requirement lists while keeping their original order
def merge_string_list(
    current: list[str] | None, incoming: list[str] | None
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    for item in (current or []) + (incoming or []):
        if item not in seen:
            merged.append(item)
            seen.add(item)

    return merged


# Merges one dictionary style field such as types constants or objects
def merge_mapping_field(
    current: dict[str, Any],
    fragment: dict[str, Any],
    key: str,
    replace: set[str],
) -> None:
    if key not in fragment:
        return
    if key in replace or current.get(key) is None:
        current[key] = deepcopy(fragment[key])
        return
    if isinstance(current.get(key), dict) and isinstance(fragment[key], dict):
        current[key] = {**current[key], **fragment[key]}
        return
    current[key] = deepcopy(fragment[key])


# Treats missing parser headings as absent sections instead of hard failures
def safe_parse(parse_fn: Any, text: str, *args: Any, **kwargs: Any) -> Any:
    try:
        return parse_fn(text, *args, **kwargs)
    except ValueError as exc:
        message = str(exc)
        if (
            "Could not find heading" in message
            or "Could not find exactly one block" in message
        ):
            return None
        raise


# Splits multi action update text into one chunk per action block
def split_action_chunks(text: str) -> list[str]:
    if "## NEXT ACTION" not in text:
        start = text.find("### Action Parameters")
        if start == -1:
            return []
        return [text[start:].strip()]

    chunks: list[str] = []
    for part in re.split(r"^\s*## NEXT ACTION\s*$", text, flags=re.MULTILINE):
        start = part.find("### Action Parameters")
        if start != -1:
            chunks.append(part.strip())
    return chunks


# Reads an action name from the text that appears before an action block
def extract_action_name(chunk: str) -> str | None:
    start = chunk.find("### Action Parameters")
    if start == -1:
        return None

    lines = [line.strip() for line in chunk[:start].splitlines() if line.strip()]
    for line in reversed(lines):
        if (
            line.startswith("#")
            or line.startswith("```")
            or line in {"[PREDICATES]", "[ACTION NAME]"}
            or line.startswith("{")
            or line.startswith("[")
            or line.startswith("(")
        ):
            continue
        return line.strip("[]")
    return None


# Parses one or more action blocks from the update text
def parse_action_updates(
    text: str,
    action_name: str | list[str] | None = None,
) -> list[dict[str, Any]]:
    chunks = split_action_chunks(text)
    if not chunks:
        return []

    if action_name is None:
        provided_names: list[str] = []
    elif isinstance(action_name, str):
        provided_names = [action_name]
    else:
        provided_names = list(action_name)
    actions: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks):
        inferred_name = extract_action_name(chunk)
        provided_name = provided_names[index] if index < len(provided_names) else None
        resolved_name = inferred_name or provided_name
        if resolved_name is None:
            raise ValueError(
                "`action_name` is required when the update includes action sections without action names in the text."
            )

        start = chunk.find("### Action Parameters")
        actions.append(parse_action(chunk[start:].strip(), resolved_name))

    if len(provided_names) > len(chunks):
        raise ValueError("More `action_name` values were provided than action updates.")

    return actions


# Merges a parsed domain fragment into the current domain state
def merge_domain_fragment(
    domain: dict[str, Any] | None,
    fragment: dict[str, Any],
    replace_fields: list[str] | None = None,
) -> dict[str, Any]:
    current = deepcopy(domain or {})
    replace = set(replace_fields or [])

    for key in ["types", "constants"]:
        merge_mapping_field(current, fragment, key, replace)

    for key in ["predicates", "functions", "actions"]:
        if key not in fragment:
            continue
        if key in replace:
            current[key] = deepcopy(fragment[key])
        else:
            current[key] = merge_named_entries(current.get(key), fragment[key])

    if "requirements" in fragment:
        if "requirements" in replace:
            current["requirements"] = list(fragment["requirements"])
        else:
            current["requirements"] = merge_string_list(
                current.get("requirements"), fragment["requirements"]
            )

    return current


# Merges a parsed task fragment into the current task state
def merge_task_fragment(
    task: dict[str, Any] | None,
    fragment: dict[str, Any],
    replace_fields: list[str] | None = None,
) -> dict[str, Any]:
    current = deepcopy(task or {})
    replace = set(replace_fields or [])

    merge_mapping_field(current, fragment, "objects", replace)

    for key in ["initial", "goal"]:
        if key not in fragment:
            continue
        normalized = normalize_states(fragment[key])
        if key in replace:
            current[key] = normalized
        else:
            current[key] = merge_states(current.get(key), normalized)

    return current


# Parses one optional domain section if its heading exists
def parse_domain_section(
    text: str,
    heading: str,
    parse_fn: Any,
    use_type_hierarchy: bool = False,
) -> Any:
    if f"### {heading}" not in text:
        return None
    resolved_parse_fn = (
        parse_type_hierarchy if heading == "TYPES" and use_type_hierarchy else parse_fn
    )
    return safe_parse(resolved_parse_fn, text)


# Parses non action domain sections such as types constants predicates and functions
def parse_domain_fragment_text(
    text: str, use_type_hierarchy: bool = False
) -> dict[str, Any]:
    fragment: dict[str, Any] = {}
    sections = [
        ("types", "TYPES", parse_types, True),
        ("constants", "CONSTANTS", parse_constants, True),
        ("predicates", "New Predicates", parse_new_predicates, False),
        ("functions", "FUNCTIONS", parse_functions, False),
    ]
    for key, heading, parse_fn, keep_empty in sections:
        parsed = parse_domain_section(
            text=text,
            heading=heading,
            parse_fn=parse_fn,
            use_type_hierarchy=use_type_hierarchy,
        )
        if parsed or (keep_empty and parsed is not None):
            fragment[key] = parsed

    return fragment


# Parses a full domain update including optional action sections
def parse_domain_update_text(
    text: str,
    action_name: str | list[str] | None = None,
    use_type_hierarchy: bool = False,
) -> dict[str, Any]:
    fragment = parse_domain_fragment_text(
        text=text,
        use_type_hierarchy=use_type_hierarchy,
    )

    actions = parse_action_updates(text, action_name)
    if actions:
        fragment["actions"] = actions

    if not fragment:
        raise ValueError(
            "No recognizable domain update sections were found in `domain_update`."
        )

    return fragment


# Parses one task state section such as INITIAL or GOAL
def parse_task_state_section(
    text: str,
    heading: str,
    parse_fn: Any,
) -> list[dict[str, Any]] | None:
    if f"### {heading}" not in text:
        return None

    parsed = safe_parse(parse_fn, text)
    if parsed:
        return parsed

    heading_text = safe_parse(parse_heading, text, heading)
    if heading_text is None:
        return None

    raw = remove_comments(combine_blocks(heading_text))
    parsed_pddl = parse_pddl(f"({raw})")
    if (
        isinstance(parsed_pddl, list)
        and len(parsed_pddl) == 1
        and isinstance(parsed_pddl[0], list)
    ):
        parsed_pddl = parsed_pddl[0]
    if isinstance(parsed_pddl, list) and parsed_pddl and parsed_pddl[0] == "and":
        return parse_task_states(parsed_pddl[1:])
    return parse_task_states(parsed_pddl)


# Parses task objects initial state and goal from task update text
def parse_task_fragment_text(text: str) -> dict[str, Any]:
    fragment: dict[str, Any] = {}
    objects = parse_domain_section(text, "OBJECTS", parse_objects)
    if objects is not None:
        fragment["objects"] = objects

    initial = parse_task_state_section(text, "INITIAL", parse_initial)
    goal = parse_task_state_section(text, "GOAL", parse_goal)
    for key, value in (("initial", initial), ("goal", goal)):
        if value is not None:
            fragment[key] = value

    return fragment


# Validates that a task update produced at least one recognized section
def parse_task_update_text(text: str) -> dict[str, Any]:
    fragment = parse_task_fragment_text(text)
    if not fragment:
        raise ValueError(
            "No recognizable task update sections were found in `task_update`."
        )
    return fragment
