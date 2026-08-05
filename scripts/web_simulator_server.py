from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from aiohttp import WSMsgType, web

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.replay_session import ReplaySession


async def create_app(args: argparse.Namespace) -> web.Application:
    max_windows = None if int(args.max_windows) <= 0 else int(args.max_windows)
    session = ReplaySession(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        start_row=args.start_row,
        max_windows=max_windows,
        ollama_model=args.model,
        ollama_base_url=args.base_url,
    )
    app = web.Application()
    app["session"] = session
    app["static_dir"] = Path(args.web_root)
    app["stream_task"] = None
    app["stream_running"] = False
    app["stream_delay"] = float(args.delay)

    async def send_state(ws: web.WebSocketResponse) -> None:
        await ws.send_json({"type": "state", "payload": session.status()})

    async def stream_loop(ws: web.WebSocketResponse) -> None:
        try:
            while not ws.closed and app["stream_running"]:
                try:
                    emitted = await session.step_row()
                except Exception as exc:
                    app["stream_running"] = False
                    await ws.send_json({"type": "error", "payload": str(exc)})
                    await send_state(ws)
                    break
                for record in emitted:
                    await ws.send_json({"type": "window", "payload": record})
                await send_state(ws)
                if session.finished:
                    app["stream_running"] = False
                    await ws.send_json({"type": "done", "payload": session.status()})
                    break
                await asyncio.sleep(app["stream_delay"])
        finally:
            app["stream_task"] = None

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)
        await send_state(ws)

        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "payload": "invalid_json"})
                continue

            msg_type = payload.get("type")
            if msg_type == "start":
                app["stream_running"] = True
                if app["stream_task"] is None:
                    app["stream_task"] = asyncio.create_task(stream_loop(ws))
            elif msg_type == "pause":
                app["stream_running"] = False
                await send_state(ws)
            elif msg_type == "resume":
                app["stream_running"] = True
                if app["stream_task"] is None:
                    app["stream_task"] = asyncio.create_task(stream_loop(ws))
                await send_state(ws)
            elif msg_type == "step":
                app["stream_running"] = False
                try:
                    emitted = await session.step_row()
                except Exception as exc:
                    await ws.send_json({"type": "error", "payload": str(exc)})
                    await send_state(ws)
                    continue
                for record in emitted:
                    await ws.send_json({"type": "window", "payload": record})
                await send_state(ws)
            elif msg_type == "reset":
                app["stream_running"] = False
                session.reset()
                await send_state(ws)
            elif msg_type == "set_delay":
                delay = float(payload.get("value", app["stream_delay"]))
                app["stream_delay"] = max(0.0, delay)
                await send_state(ws)
            else:
                await ws.send_json({"type": "error", "payload": f"unknown_type:{msg_type}"})

        app["stream_running"] = False
        return ws

    async def index(request: web.Request) -> web.StreamResponse:
        index_path = app["static_dir"] / "index.html"
        if index_path.exists():
            return web.FileResponse(index_path)
        return web.Response(
            text="Build the frontend first with `npm.cmd install` and `node node_modules/vite/bin/vite.js build`.",
            content_type="text/plain",
        )

    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/health", lambda request: web.json_response({"ok": True}))
    app.router.add_get("/", index)
    assets_dir = app["static_dir"] / "assets"
    if assets_dir.exists():
        app.router.add_static("/assets/", path=assets_dir, show_index=False)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_mamba_watcher.pt")
    parser.add_argument("--web-root", default="web/dist")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--base-url", default="http://localhost:11434")
    args = parser.parse_args()

    app = asyncio.run(create_app(args))
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
