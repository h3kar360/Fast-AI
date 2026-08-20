from fastapi import FastAPI
from app.api.routes import router
from app.db import engine, Base

from app.api.model import Chats

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title = "Fast AI",
    description = "A simple API using OpenAI API",
    version = "1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

app.include_router(router)