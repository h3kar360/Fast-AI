from pydantic import BaseModel

class CreateChat(BaseModel):
    title: str

class UpdateContext(BaseModel):
    chat_context: str

class ChatInput(BaseModel):
    message: str

class ChatResponse(BaseModel):
    id: int
    title: str
    response: str

    class Config:
        from_attributes = True

class ChatInfo(BaseModel):
    id: int
    title: str