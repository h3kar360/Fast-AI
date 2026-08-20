from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from app.api.model import Chats
from app.api.schema import CreateChat, UpdateContext

async def create_chat(db: AsyncSession, chat_info: CreateChat) -> Chats:
    db_chat = Chats(**chat_info.model_dump()) # model_dump: turning objects ({...}) to allow class instances (**)
    db.add(db_chat)
    await db.commit()
    await db.refresh(db_chat)
    return db_chat

async def get_all_chats(db: AsyncSession) -> list[Chats]:
    result = await db.execute(
        select(Chats)
    )
    return list(result.scalars().all())

async def get_chat_by_id(db: AsyncSession, chat_id: int) -> Chats | None:
    return await db.get(Chats, chat_id)

async def update_context(db: AsyncSession, updated_context: str , chat_id: int) -> Chats | None:
    update_data = {
        "chat_context": updated_context
    }

    result = await db.execute((
        update(Chats)
        .where(Chats.id == chat_id)
        .values(**update_data)
        .returning(Chats)
    ))

    await db.commit()
    return result.scalar_one_or_none() # converts tuple from postgresql to an object

async def delete_chat(db: AsyncSession, chat_id: int) -> bool:
    result = await db.execute((
        delete(Chats)
        .where(Chats.id == chat_id)
        .returning(Chats.id)
    ))

    await db.commit()
    return result.scalar_one_or_none() is not None
