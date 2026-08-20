from app.db import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP, Boolean, text

class Chats(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index = True, nullable = False)
    title = Column(String, nullable = False)
    chat_context = Column(String, nullable = True)
    created_at = Column(TIMESTAMP(timezone = True), server_default = text('now()'))