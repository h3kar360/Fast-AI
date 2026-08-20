from sqlalchemy.orm import Session
from app.api.model import Chats
from app.api.schema import CreateChat, UpdateContext

def create_chat(db: Session, chat_info: CreateChat):
    db_chat = Chats(**chat_info.model_dump()) # model_dump: turning objects ({...}) to allow class instances (**)
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat

def get_all_chats(db: Session):
    return db.query(Chats).all()

def get_chat_by_id(db: Session, chat_id: int):
    return db.query(Chats).filter(Chats.id == chat_id).first()

def update_context(db: Session, chat_context: UpdateContext, chat_id: int):
    db_chat = get_chat_by_id(db, chat_id)
    if not db_chat:
        return None

    update_data = chat_context.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_chat, key, value)

    db.commit()
    db.refresh(db_chat)
    return db_chat

def delete_chat(db: Session, chat_id: int):
    db_chat = get_chat_by_id(db, chat_id)
    if not db_chat:
        return None

    db.delete(db_chat)
    db.commit()
    return db_chat
