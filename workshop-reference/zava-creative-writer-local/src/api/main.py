from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from orchestrator import Task, create

base = Path(__file__).resolve().parent
ui_dir = base.parent.parent / "ui"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Local-only workshop demo — allow any origin for localhost testing
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/article")
async def create_article(task: Task):
    return StreamingResponse(
        create(task.research, task.products, task.assignment),
        media_type="text/event-stream",
    )


# Serve the shared UI as static files (must be mounted last so the
# /api route takes priority).
if ui_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
