from app.db import Base
from sqlalchemy import Column, ForeignKey, Integer, String, TIMESTAMP, text
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship

class Chats(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    chat_context = Column(String, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default = text('now()'))

class Documents(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))

    # One-to-Many Relationship: One Document has many Embeddings/Chunks
    embeddings = relationship(
        "Embeddings", 
        back_populates="document", 
        cascade="all, delete-orphan"
    )

class Embeddings(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True, nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content = Column(String, nullable=False)
    embedding = Column(Vector(2048), nullable=False)

    document = relationship("Documents", back_populates="embeddings")