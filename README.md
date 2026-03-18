# l2p-mcp

Standalone MCP server that wraps deterministic parts of the external [`l2p`](https://github.com/AI-Planning/l2p) package.

This draft is aimed at human-in-the-loop planning workflows where the MCP client already has model access. The server does not build or call an LLM. Instead, it helps the client:

- parse partial domain and task fragments from model output
- merge those fragments into evolving structured state
- generate final PDDL domain and problem files

## Files

- `server.py`: MCP tools and server entrypoint
- `server_helpers.py`: parsing, merge, and normalization helpers used by the MCP tools

## Tool Surface

The server currently exposes eight MCP tools:

- `parse_domain_fragment`
- `parse_action_fragment`
- `parse_task_fragment`
- `merge_domain`
- `merge_task`
- `generate_requirements`
- `generate_domain`
- `generate_task`

## Intended Workflow

1. The MCP client's model drafts a partial domain or task fragment.
2. The server parses that text into structured data.
3. The client merges the new fragment into its current domain/task state.
4. The server generates final PDDL when needed.

This supports incremental edits, so an agent or human can modify only the goal, one action, a few predicates, and so on without restarting the whole formulation process.

## Examples

`parse_domain_fragment`
: Parse headings like `### TYPES`, `### CONSTANTS`, `### New Predicates`, and `### FUNCTIONS` from model output.

`parse_action_fragment`
: Parse a single action from `### Action Parameters`, `### Action Preconditions`, and `### Action Effects`.

`parse_task_fragment`
: Parse headings like `### OBJECTS`, `### INITIAL`, and `### GOAL`.

`merge_domain` / `merge_task`
: Merge a partial fragment into the current structured state. Use `replace_fields` when a field such as `goal` should be replaced instead of appended/upserted.

`generate_domain` / `generate_task`
: Produce final PDDL strings from the current structured state.

## Requirements

- An environment where `l2p` is installed in the same interpreter used to run the server
- `mcp`

## Run

```bash
python3 server.py
```

## Example MCP client config

```json
{
  "mcpServers": {
    "l2p": {
      "command": "python3",
      "args": ["/absolute/path/to/server.py"]
    }
  }
}
```
