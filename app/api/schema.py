from pydantic import BaseModel, ConfigDict
from typing import Optional

# ---Chats---

class CreateChat(BaseModel):
    title: str

class UpdateContext(BaseModel):
    chat_context: str

class ChatInput(BaseModel):
    message: str

class ChatResponse(BaseModel):
    id: int
    title: str
    response: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class ChatInfo(BaseModel):
    id: int
    title: str

    model_config = ConfigDict(from_attributes=True)

# ---Docs---

class CreateDocs(BaseModel):
    title: str

class DocsCreatedResponse(BaseModel):
    id: int
    title: str

    model_config = ConfigDict(from_attributes=True)
    