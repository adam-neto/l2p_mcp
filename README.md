# l2p-mcp

Standalone MCP server that wraps the external [`l2p`](https://github.com/AI-Planning/l2p) package.

Files submitted in this repo:

- `server.py`: the MCP server and all tool logic
- `test_offline.py`: the offline unit test suite

The project does not vendor or clone `l2p`. It imports `l2p` at runtime and fails clearly if the package is missing.

## Tools

The server exposes three MCP tools:

- `formalize_domain_predicates`
- `formalize_task`
- `generate_task`

## Requirements

- Python 3.10+
- An environment where `l2p` can be installed
- `llm` and any provider plugin you want to use through `l2p.llm.UnifiedLLM`
- An API key only if your chosen provider requires one

## Install

Install `l2p`, `llm`, and any provider plugin separately in your environment before running the wrapper.

## Run

Run the MCP server over `stdio`:

```bash
python3 server.py
```

## Example MCP client config

```json
{
  "mcpServers": {
    "l2p": {
      "command": "python3",
      "args": ["/absolute/path/to/server.py"],
      "env": {
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

## Tool behavior

`formalize_domain_predicates`
: Uses `l2p.DomainBuilder.formalize_predicates` to infer predicates from a natural-language domain description and prompt template. The server wraps models with `l2p.llm.UnifiedLLM`, so model selection is provider-agnostic.

`formalize_task`
: Uses `l2p.TaskBuilder.formalize_task` to convert a natural-language task description into structured PDDL task data.

`generate_task`
: Uses `l2p.TaskBuilder.generate_task` to emit a PDDL problem string from structured task components.

Both LLM-backed tools accept `provider`, `model`, `api_key`, `api_key_env`, and `config_path` so they can target any provider configured in `l2p`'s `llm.yaml`.

## Development

Basic syntax check:

```bash
python3 -m py_compile server.py test_offline.py
```

## Testing

Offline suite:

```bash
python3 -m unittest -v test_offline
```

Current offline status: all 8 offline tests pass.
