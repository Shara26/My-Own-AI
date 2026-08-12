

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


FRONTEND_FILE = Path(__file__).resolve().parent / "index.html"


def create_server(title: str = "My Own AI") -> FastAPI:
   
    app = FastAPI(title=title, description="An educational Vector Database with HNSW, KD-Tree, Brute Force search and a local-LLM RAG pipeline.")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def serve_index() -> FileResponse:
        if not FRONTEND_FILE.exists():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(FRONTEND_FILE, media_type="text/html")

    return app


def run_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8080) -> None:
  
    uvicorn.run(app, host=host, port=port)
