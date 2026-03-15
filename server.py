from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


def _read_text(text: str | None, path: str | None, field_name: str) -> str:
    if text:
        return text
    if path:
        return Path(path).expanduser().read_text(encoding="utf-8")
    raise ValueError(f"`{field_name}` or `{field_name}_path` is required.")


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {key: _serialize(item) for key, item in vars(value).items()}
    return value


def _normalize_states(
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


def _build_llm(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    config_path: str | None = None,
) -> Any:
    try:
        import l2p
        from l2p.llm.unified import UnifiedLLM
    except ImportError as exc:
        raise RuntimeError(
            "The `l2p` package is not installed in the active Python environment."
        ) from exc

    return UnifiedLLM(
        provider=provider,
        model=model,
        config_path=config_path
        or str(Path(l2p.__file__).resolve().parent / "llm" / "utils" / "llm.yaml"),
        api_key=api_key or os.getenv(api_key_env),
    )


mcp = FastMCP(
    name="l2p",
    instructions="Wraps the external l2p package for PDDL domain and task formalization plus task generation.",
)


@mcp.tool(
    description="Infer PDDL predicates from a domain description using l2p.DomainBuilder."
)
def formalize_domain_predicates(
    domain_description: str | None = None,
    domain_description_path: str | None = None,
    prompt_template: str | None = None,
    prompt_template_path: str | None = None,
    types: dict | list[dict] | None = None,
    constants: dict | list[dict] | None = None,
    predicates: list[dict] | None = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    config_path: str | None = None,
) -> dict:
    from l2p import DomainBuilder, Predicate

    result = DomainBuilder().formalize_predicates(
        model=_build_llm(provider, model, api_key, api_key_env, config_path),
        domain_desc=_read_text(
            domain_description, domain_description_path, "domain_description"
        ),
        prompt_template=_read_text(
            prompt_template, prompt_template_path, "prompt_template"
        ),
        types=types,
        constants=constants,
        predicates=None if predicates is None else [Predicate(**p) for p in predicates],
        functions=None,
    )
    return _serialize(result)


@mcp.tool(description="Extract a PDDL task from a natural-language problem description.")
def formalize_task(
    problem_description: str | None = None,
    problem_description_path: str | None = None,
    prompt_template: str | None = None,
    prompt_template_path: str | None = None,
    types: dict | list[dict] | None = None,
    predicates: list[dict] | None = None,
    constants: dict | list[dict] | None = None,
    functions: list[dict] | None = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    config_path: str | None = None,
) -> dict:
    from l2p import Function, Predicate, TaskBuilder

    result = TaskBuilder().formalize_task(
        model=_build_llm(provider, model, api_key, api_key_env, config_path),
        problem_desc=_read_text(
            problem_description, problem_description_path, "problem_description"
        ),
        prompt_template=_read_text(
            prompt_template, prompt_template_path, "prompt_template"
        ),
        types=types,
        predicates=None if predicates is None else [Predicate(**p) for p in predicates],
        constants=constants,
        functions=None if functions is None else [Function(**f) for f in functions],
    )
    return _serialize(result)


@mcp.tool(description="Generate a PDDL problem string from structured task components.")
def generate_task(
    domain_name: str,
    problem_name: str,
    objects: dict[str, str] | None = None,
    initial: list[str] | list[dict] | None = None,
    goal: list[str] | list[dict] | None = None,
    metric: str | None = None,
) -> dict:
    from l2p import TaskBuilder

    if metric is not None:
        raise ValueError("`metric` is not supported by l2p.TaskBuilder.generate_task.")
    if objects is None or not isinstance(objects, dict):
        raise ValueError("`objects` is required and must be a dict[str, str].")
    if initial is None or not all(isinstance(item, dict) for item in initial):
        raise ValueError("`initial` is required and must be a list of dicts.")
    if goal is None or not all(isinstance(item, dict) for item in goal):
        raise ValueError("`goal` is required and must be a list of dicts.")

    task = TaskBuilder().generate_task(
        domain_name=domain_name,
        problem_name=problem_name,
        objects=objects,
        initial=_normalize_states(initial),
        goal=_normalize_states(goal),
    )
    return {"task": _serialize(task)}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
