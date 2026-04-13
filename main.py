import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.db.database import init_db
from backend.routes.ws_routes import router as ws_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    init_db()
    os.makedirs("static/recordings", exist_ok=True)
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/frames", exist_ok=True)
    yield
    # Shutdown logic (optional)

app = FastAPI(title="AI Vision Intelligence Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)

# Serve media files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the build artifacts
if os.path.exists("frontend/dist"):

    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")

else:
    @app.get("/")
    async def root():
        return {"message": "Backend OK. Frontend build not found."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
