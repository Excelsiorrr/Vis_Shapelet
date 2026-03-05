from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from backend.api.part_a import router as part_a_router
from backend.core.constants import SPEC_VERSION


def create_app() -> FastAPI:
    app = FastAPI(title="ShapeX API", version=SPEC_VERSION)
    app.include_router(part_a_router)

    @app.get("/", include_in_schema=False)
    def debug_home() -> FileResponse:
        return FileResponse(Path(__file__).with_name("part_a_debug.html"))

    @app.get("/part-a-debug", include_in_schema=False)
    def part_a_debug_page() -> FileResponse:
        return FileResponse(Path(__file__).with_name("part_a_debug.html"))

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok", "version": SPEC_VERSION}

    return app


app = create_app()
