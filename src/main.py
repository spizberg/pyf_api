"""FastAPI main module."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .model_utils import lifespan

from .routers import v1_router, v2_router


app = FastAPI(title="PrintYourFeet API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
@app.get("/api", tags=["Health Check"])
async def healthz():
    return "Hello on Print Your Feet!"


app.include_router(router=v1_router)
app.include_router(router=v2_router)
