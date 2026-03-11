from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from backend.api.part_a import router as part_a_router
from backend.api.part_b import router as part_b_router
from backend.api.part_c import router as part_c_router
from backend.api.part_e import router as part_e_router
from backend.core.constants import SPEC_VERSION
from fastapi.middleware.cors import CORSMiddleware

def create_app() -> FastAPI:
    app = FastAPI(title="ShapeX API", version=SPEC_VERSION)
    app.include_router(part_a_router)
    app.include_router(part_b_router)
    app.include_router(part_c_router)
    app.include_router(part_e_router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.get("/", include_in_schema=False)
    def debug_home() -> FileResponse:
        return FileResponse(Path(__file__).with_name("part_a_debug.html"))

    @app.get("/part-a-debug", include_in_schema=False)
    def part_a_debug_page() -> FileResponse:
        return FileResponse(Path(__file__).with_name("part_a_debug.html"))

    @app.get("/part-b-debug", include_in_schema=False)
    def part_b_debug_page() -> FileResponse:
        return FileResponse(Path(__file__).with_name("part_b_debug.html"))

    @app.get("/part-c-debug", include_in_schema=False)
    def part_c_debug_page() -> FileResponse:
        return FileResponse(Path(__file__).with_name("part_c_debug.html"))

    @app.get("/part-e-debug", include_in_schema=False)
    def part_e_debug_page() -> FileResponse:
        return FileResponse(Path(__file__).with_name("part_e_debug.html"))

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok", "version": SPEC_VERSION}

    return app


app = create_app()
