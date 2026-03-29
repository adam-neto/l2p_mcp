FROM python:3.14-slim

ENV MCP_TRANSPORT=streamable-http
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8002
ENV MCP_STREAMABLE_HTTP_PATH=/mcp

WORKDIR /app

COPY server.py /app/server.py
COPY server_helpers.py /app/server_helpers.py

RUN pip install --no-cache-dir mcp l2p

EXPOSE 8002

CMD ["python3", "server.py"]
