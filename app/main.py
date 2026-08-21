from contextlib import asynccontextmanager
from sqlalchemy import text
from fastapi import FastAPI
from app.api.routes import router
from app.db import engine, Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()

app = FastAPI(
    title="Fast AI",
    description="A simple API using OpenAI API",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Hello World"}