"""FastAPI server for remote monitoring and interaction.

Run on the Lambda server:
    uv run python -m minsky.server --host 0.0.0.0 --port 8000

Connect from local laptop to monitor and interact with the AI system.
"""

import asyncio
import json
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from minsky.orchestrator import Orchestrator
from minsky.types import Message, RoomType, MessageType


# Global state
orchestrator: Orchestrator | None = None
connected_clients: list[WebSocket] = []
message_history: list[dict] = []
is_running: bool = False


async def broadcast(data: dict) -> None:
    """Broadcast message to all connected WebSocket clients."""
    message = json.dumps(data)
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_text(message)
        except:
            disconnected.append(client)
    for client in disconnected:
        connected_clients.remove(client)


def on_message(msg: Message) -> None:
    """Callback for messages - broadcasts to clients."""
    data = {
        "type": "message",
        "source": msg.source.value,
        "target": msg.target.value,
        "message_type": msg.message_type.value,
        "content": msg.content,
        "cycle": msg.cycle,
        "timestamp": datetime.now().isoformat(),
    }
    message_history.append(data)
    asyncio.create_task(broadcast(data))


def on_cycle_start(cycle: int) -> None:
    """Callback for cycle start."""
    data = {
        "type": "cycle_start",
        "cycle": cycle,
        "timestamp": datetime.now().isoformat(),
    }
    asyncio.create_task(broadcast(data))


def on_cycle_end(cycle: int, outputs: list[Message]) -> None:
    """Callback for cycle end."""
    data = {
        "type": "cycle_end",
        "cycle": cycle,
        "outputs": [o.content for o in outputs],
        "timestamp": datetime.now().isoformat(),
    }
    asyncio.create_task(broadcast(data))


def on_summarize(room: str, summary: str) -> None:
    """Callback for summarization."""
    data = {
        "type": "summary",
        "room": room,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }
    asyncio.create_task(broadcast(data))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize orchestrator on startup."""
    global orchestrator
    orchestrator = Orchestrator(
        max_cycles=100,
        summarizer_interval=10,
        on_message=on_message,
        on_cycle_start=on_cycle_start,
        on_cycle_end=on_cycle_end,
        on_summarize=on_summarize,
    )
    yield
    # Cleanup


app = FastAPI(title="Minsky Society of Mind", lifespan=lifespan)

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "running": is_running,
        "global_step": orchestrator.current_cycle if orchestrator else 0,
    }


@app.get("/status")
async def status():
    """Get current system status."""
    if not orchestrator:
        return {"error": "Orchestrator not initialized"}

    return {
        "global_step": orchestrator.current_cycle,
        "running": is_running,
        "use_llm": orchestrator.use_llm,
        "use_edit": orchestrator.use_edit,
        "use_summarizers": orchestrator.use_summarizers,
        "queue_size": len(orchestrator.message_queue),
        "summaries": orchestrator.room_summaries,
    }


@app.post("/init")
async def init_models(use_models: bool = True, rwkv_path: str | None = None):
    """Initialize RWKV and T5 models."""
    global orchestrator

    if not orchestrator:
        return {"error": "Orchestrator not initialized"}

    try:
        # Initialize RWKV
        from minsky.llm_client import RWKVConfig
        config = RWKVConfig()
        if rwkv_path:
            config.model_path = rwkv_path
        orchestrator.rwkv.initialize(config)

        orchestrator.use_llm = True
        orchestrator.use_summarizers = True

        # Initialize T5 if using full stack
        if use_models:
            orchestrator.t5_edit.initialize()
            orchestrator.use_edit = True

        return {"status": "ok", "models_loaded": True}
    except Exception as e:
        return {"error": str(e)}


@app.post("/send")
async def send_message(content: str):
    """Send a message to the system."""
    global is_running

    if not orchestrator:
        return {"error": "Orchestrator not initialized"}

    # Inject message
    orchestrator.inject_message(Message(
        content=content,
        source=RoomType.EXTERNAL,
        target=RoomType.SENSORY,
        message_type=MessageType.PERCEPTION,
    ))

    orchestrator.inject_message(Message(
        content=content,
        source=RoomType.SENSORY,
        target=RoomType.PLANNING,
        message_type=MessageType.PERCEPTION,
    ))

    return {"status": "ok", "message": "Message injected"}


@app.post("/step")
async def run_step():
    """Run a single global step."""
    if not orchestrator:
        return {"error": "Orchestrator not initialized"}

    outputs = orchestrator.run_cycle()

    return {
        "global_step": orchestrator.current_cycle,
        "outputs": [o.content for o in outputs],
    }


@app.post("/run")
async def run_until_output(max_steps: int = 10):
    """Run until output or max steps."""
    global is_running

    if not orchestrator:
        return {"error": "Orchestrator not initialized"}

    if is_running:
        return {"error": "Already running"}

    is_running = True
    all_outputs = []

    try:
        for _ in range(max_steps):
            outputs = orchestrator.run_cycle()
            all_outputs.extend(outputs)

            if outputs and not orchestrator.message_queue:
                break
            if not orchestrator.message_queue:
                break

            # Yield to event loop for WebSocket updates
            await asyncio.sleep(0.01)
    finally:
        is_running = False

    return {
        "global_step": orchestrator.current_cycle,
        "outputs": [o.content for o in all_outputs],
    }


@app.get("/history")
async def get_history(limit: int = 100):
    """Get message history."""
    return {"messages": message_history[-limit:]}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    connected_clients.append(websocket)

    # Send current status
    await websocket.send_json({
        "type": "connected",
        "global_step": orchestrator.current_cycle if orchestrator else 0,
    })

    try:
        while True:
            # Keep connection alive, handle incoming commands
            data = await websocket.receive_text()
            try:
                cmd = json.loads(data)
                if cmd.get("action") == "send":
                    await send_message(cmd.get("content", ""))
                elif cmd.get("action") == "step":
                    await run_step()
                elif cmd.get("action") == "run":
                    await run_until_output(cmd.get("max_steps", 10))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


# Simple HTML frontend served from the server
FRONTEND_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Minsky Society of Mind</title>
    <style>
        body { font-family: monospace; background: #1a1a2e; color: #eee; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00d9ff; }
        .status { background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .messages { background: #0f0f23; padding: 15px; border-radius: 8px; height: 400px; overflow-y: auto; }
        .message { padding: 5px 10px; margin: 5px 0; border-left: 3px solid #444; }
        .message.sensory { border-color: #00ff88; }
        .message.planning { border-color: #ff8800; }
        .message.motor { border-color: #ff0088; }
        .message.external { border-color: #00d9ff; }
        .input-area { margin-top: 20px; display: flex; gap: 10px; }
        input { flex: 1; padding: 10px; background: #16213e; border: 1px solid #444; color: #eee; border-radius: 4px; }
        button { padding: 10px 20px; background: #00d9ff; color: #000; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #00b8d9; }
        .summaries { background: #16213e; padding: 15px; border-radius: 8px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Minsky Society of Mind</h1>

        <div class="status" id="status">
            Connecting...
        </div>

        <div class="messages" id="messages"></div>

        <div class="input-area">
            <input type="text" id="input" placeholder="Send a message to the system..." />
            <button onclick="sendMessage()">Send</button>
            <button onclick="runStep()">Step</button>
            <button onclick="runUntilOutput()">Run</button>
        </div>

        <div class="summaries" id="summaries">
            <h3>Room Summaries</h3>
            <div id="summary-content">No summaries yet.</div>
        </div>
    </div>

    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        const messagesDiv = document.getElementById('messages');
        const statusDiv = document.getElementById('status');
        const summaryDiv = document.getElementById('summary-content');

        ws.onopen = () => {
            statusDiv.textContent = 'Connected';
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'message') {
                const div = document.createElement('div');
                div.className = `message ${data.source}`;
                div.textContent = `[Step ${data.cycle}] ${data.source} → ${data.target}: ${data.content.slice(0, 200)}`;
                messagesDiv.appendChild(div);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            } else if (data.type === 'cycle_start') {
                statusDiv.textContent = `Global Step ${data.cycle}`;
            } else if (data.type === 'summary') {
                summaryDiv.innerHTML += `<p><strong>${data.room}:</strong> ${data.summary}</p>`;
            }
        };

        ws.onclose = () => {
            statusDiv.textContent = 'Disconnected';
        };

        function sendMessage() {
            const input = document.getElementById('input');
            ws.send(JSON.stringify({action: 'send', content: input.value}));
            input.value = '';
        }

        function runStep() {
            ws.send(JSON.stringify({action: 'step'}));
        }

        function runUntilOutput() {
            ws.send(JSON.stringify({action: 'run', max_steps: 20}));
        }

        document.getElementById('input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Serve the web UI."""
    return FRONTEND_HTML


def main():
    """Run the server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Minsky Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Starting Minsky server at http://{args.host}:{args.port}")
    print(f"Web UI: http://{args.host}:{args.port}/ui")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
