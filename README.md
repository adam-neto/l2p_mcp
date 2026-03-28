# l2p-mcp

Standalone MCP server that wraps deterministic parts of the external [`l2p`](https://github.com/AI-Planning/l2p) package for human-in-the-loop planning workflows.

This repo assumes the MCP client already has model access. The model only needs to decide whether a change belongs to the domain or the task, then format the update using the tool docstring. After that, the server takes over: it parses the update, merges it into the current planning state, infers requirements when needed, and generates updated PDDL artifacts.

## Files

### `server/`

- `server.py`
  Defines the MCP tools, configures the HTTP transport defaults, and starts the server
- `server_helpers.py`
  Contains the parsing, merge, and normalization logic used by the MCP tools
- `Dockerfile`
  Builds the Docker image for running the MCP server as a local service
- `docker-compose.yml`
  Defines the local Docker Compose service that exposes the MCP server on `localhost:8001`
- `.dockerignore`
  Keeps the Docker build context small by excluding caches and local development files

### `tests/`

- `test_offline.py`
  Direct unit tests for parsing, merging, generation, and error handling
- `test_mcp_offline.py`
  End-to-end MCP tests that exercise the server over a temporary `stdio` client using the active Python interpreter

## Tool Surface

The server exposes two high-level MCP tools:

- `update_domain`
- `update_task`

## What `update_domain` Does

`update_domain` creates or updates domain state from model-written text. It supports:

- `### TYPES`
- `### CONSTANTS`
- `### New Predicates`
- `### FUNCTIONS`
- zero, one, or many action updates using:
  - `### Action Parameters`
  - `### Action Preconditions`
  - `### Action Effects`
  - optional `## NEXT ACTION` separators

The tool can:

- start from an empty domain when no prior `domain` is provided
- merge into an existing domain when `domain` is provided
- infer `:requirements`
- generate final domain PDDL when `domain_name` is provided

For multi-action updates, action names can come from:

- text placed before each action block
- `action_name` passed as a list

## What `update_task` Does

`update_task` creates or updates task/problem state from model-written text. It supports:

- `### OBJECTS`
- `### INITIAL`
- `### GOAL`

Each section can contain zero, one, or many entries. The tool can:

- start from an empty task when no prior `task` is provided
- merge into an existing task when `task` is provided
- replace fields like `goal` with `replace_fields`
- generate final problem PDDL when `domain_name` and `problem_name` are provided

Task generation requires the merged task to contain:

- `objects`
- `initial`
- `goal`

## Intended Workflow

1. The user asks for a planning change
2. The client-side model chooses `update_domain` or `update_task`
3. The model writes the update using the format documented in the selected tool docstring
4. The MCP server parses the update, merges it into the current structured state, and returns updated PDDL when enough information is present

This keeps the server stateless while still supporting incremental edits. A client can update only the goal, one action, several actions, a few predicates, or any other partial fragment without restarting the whole formulation process.

## Tests

Run the direct and MCP integration tests from the root directory with:

```bash
python3 -m unittest -v tests.test_offline tests.test_mcp_offline
```

The current test suite covers:

- domain creation and updates
- task creation and updates
- multi-action parsing and merging
- empty action sections
- type hierarchy parsing
- file-based update input
- MCP tool calls
- domain and task error paths

## Docker

This repo supports running the MCP server as a long-lived local Docker service over streamable HTTP. The Docker files live in the [`server/`](server/) directory.

The container is configured to:

- use `streamable-http`
- listen on `0.0.0.0:8001` inside the container
- expose the MCP endpoint at `/mcp`
- restart automatically with `unless-stopped`

Run the Dockerized server from the repo root with:

```bash
docker compose -f server/docker-compose.yml up -d --build
```

That starts the container in the background. Once it is running, the MCP endpoint is available at:

```text
http://localhost:8001/mcp
```

Helpful commands:

- `docker compose -f server/docker-compose.yml logs -f`
- `docker compose -f server/docker-compose.yml ps`
- `docker compose -f server/docker-compose.yml restart`
- `docker compose -f server/docker-compose.yml down`

For the container to come back after a reboot, make sure Docker Desktop is configured to start automatically when you log in.

## Requirements

- An interpreter where `l2p` is installed
- `mcp`
- `python3` in the current environment

## Run

```bash
python3 server/server.py
```

By default, that now starts the server with `streamable-http` on:

```text
http://127.0.0.1:8001/mcp
```

To override the host port or path:

```bash
MCP_TRANSPORT=streamable-http MCP_HOST=127.0.0.1 MCP_PORT=8000 python3 server/server.py
```

To force the older `stdio` transport for a client that wants to spawn the process directly:

```bash
MCP_TRANSPORT=stdio python3 server/server.py
```

## Example MCP Client Config

```json
{
  "mcpServers": {
    "l2p": {
      "command": "python3",
      "args": ["/absolute/path/to/server/server.py"],
      "env": {
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

If your MCP client supports HTTP transports, point it at `http://localhost:8001/mcp` when the Docker container is running, or `http://127.0.0.1:8001/mcp` when you run the server directly.
