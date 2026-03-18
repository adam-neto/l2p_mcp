from __future__ import annotations

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
    parse_pddl,
    parse_task_states,
    parse_type_hierarchy,
    parse_types,
)
from pathlib import Path
import re
from typing import Any


def load_text(text: str | None, path: str | None, field_name: str) -> str:
    if text:
        source = text
    if path:
        source = Path(path).expanduser().read_text(encoding="utf-8")
    if not text and not path:
        raise ValueError(f"`{field_name}` or `{field_name}_path` is required.")
    return re.sub(r"```[a-zA-Z0-9_-]+\n", "```\n", source)


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


def coalesce_fields(
    base: dict[str, Any] | None, **overrides: Any
) -> dict[str, Any]:
    result = dict(base or {})
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    return result


def merge_domain_fragment(
    domain: dict[str, Any] | None,
    fragment: dict[str, Any],
    replace_fields: list[str] | None = None,
) -> dict[str, Any]:
    current = deepcopy(domain or {})
    replace = set(replace_fields or [])

    for key in ["types", "constants"]:
        if key not in fragment:
            continue
        if key in replace or current.get(key) is None:
            current[key] = deepcopy(fragment[key])
        elif isinstance(current.get(key), dict) and isinstance(fragment[key], dict):
            merged = dict(current[key])
            merged.update(fragment[key])
            current[key] = merged
        else:
            current[key] = deepcopy(fragment[key])

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


def merge_task_fragment(
    task: dict[str, Any] | None,
    fragment: dict[str, Any],
    replace_fields: list[str] | None = None,
) -> dict[str, Any]:
    current = deepcopy(task or {})
    replace = set(replace_fields or [])

    if "objects" in fragment:
        if "objects" in replace or current.get("objects") is None:
            current["objects"] = deepcopy(fragment["objects"])
        else:
            merged_objects = dict(current.get("objects", {}))
            merged_objects.update(fragment["objects"])
            current["objects"] = merged_objects

    for key in ["initial", "goal"]:
        if key not in fragment:
            continue
        normalized = normalize_states(fragment[key])
        if key in replace:
            current[key] = normalized
        else:
            current[key] = merge_states(current.get(key), normalized)

    return current


def parse_domain_fragment_text(
    text: str, use_type_hierarchy: bool = False
) -> dict[str, Any]:
    fragment: dict[str, Any] = {}
    types = (
        safe_parse(parse_type_hierarchy, text)
        if use_type_hierarchy
        else safe_parse(parse_types, text)
    )
    if types is not None:
        fragment["types"] = types

    constants = safe_parse(parse_constants, text)
    if constants is not None:
        fragment["constants"] = constants

    predicates = safe_parse(parse_new_predicates, text)
    if predicates:
        fragment["predicates"] = predicates

    functions = safe_parse(parse_functions, text)
    if functions:
        fragment["functions"] = functions

    return fragment


def parse_task_fragment_text(text: str) -> dict[str, Any]:
    def parse_states_with_and_fallback(heading: str) -> list[dict[str, Any]] | None:
        parsed = safe_parse(parse_initial if heading == "INITIAL" else parse_goal, text)
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
        if (
            isinstance(parsed_pddl, list)
            and parsed_pddl
            and parsed_pddl[0] == "and"
        ):
            return parse_task_states(parsed_pddl[1:])
        return parse_task_states(parsed_pddl)

    fragment: dict[str, Any] = {}
    objects = safe_parse(parse_objects, text)
    if objects is not None:
        fragment["objects"] = objects

    initial = parse_states_with_and_fallback("INITIAL")
    if initial is not None:
        fragment["initial"] = initial

    goal = parse_states_with_and_fallback("GOAL")
    if goal is not None:
        fragment["goal"] = goal

    return fragment
