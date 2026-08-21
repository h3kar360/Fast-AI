from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from app.api.model import Chats, Documents, Embeddings
from app.api.schema import CreateChat, CreateDocs
from app.config import client

# ---Chats---

# Create
async def create_chat(db: AsyncSession, chat_info: CreateChat) -> Chats:
    db_chat = Chats(**chat_info.model_dump()) # model_dump: turning objects ({...}) to allow class instances (**)
    db.add(db_chat)
    await db.commit()
    await db.refresh(db_chat)
    return db_chat

# Read all
async def get_all_chats(db: AsyncSession) -> list[Chats]:
    result = await db.execute(select(Chats))
    return list(result.scalars().all())

# Read
async def get_chat_by_id(db: AsyncSession, chat_id: int) -> Chats | None:
    return await db.get(Chats, chat_id)

# Update
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

# Delete
async def delete_chat(db: AsyncSession, chat_id: int) -> bool:
    result = await db.execute((
        delete(Chats)
        .where(Chats.id == chat_id)
        .returning(Chats.id)
    ))

    await db.commit()
    return result.scalar_one_or_none() is not None

# ---Docs---

# Create
async def create_doc(db: AsyncSession, doc_info: CreateDocs) -> Documents:
    db_doc = Documents(**doc_info.model_dump())
    db.add(db_doc)
    await db.commit()
    await db.refresh(db_doc)
    return db_doc

# Read All
async def get_all_docs(db: AsyncSession) -> List[Documents]:
    result = await db.execute(select(Documents))
    return list(result.scalars().all())

# Delete
async def delete_docs(db: AsyncSession, doc_id: int) -> bool:
    result = await db.execute((
            delete(Documents)
            .where(Documents.id == doc_id)
            .returning(Documents.id)
        ))
    
    await db.commit()
    return result.scalar_one_or_none() is not None

# ---Embeddings---

# Create
async def add_chunks(db: AsyncSession, chunk: str, doc_id: int) -> Embeddings:
    embedding = await client.embeddings.create(
                                    model="nvidia/nemotron-3-embed-1b:free",
                                    input=chunk,
                                    encoding_format="float"
                                )

    embedding_info = {
        "document_id": doc_id,
        "content": chunk,
        "embedding": embedding.data[0].embedding
    }

    db_embedding = Embeddings(**embedding_info)
    db.add(db_embedding)
    await db.commit()
    await db.refresh(db_embedding)
    return db_embedding

# Get closest embeddings
async def retrieve_most_similar(db: AsyncSession, query: list[float]) -> list[Embeddings]:
    return (await db.scalars(
        select(Embeddings)
        .order_by(Embeddings.embedding.cosine_distance(query))
        .limit(5)
    )).all()

# Get chunk based on closest embeddings
async def retrieve_relevant_chunks(db: AsyncSession, query: str) -> list[str]:
    embedding = await client.embeddings.create(
                                        model="nvidia/nemotron-3-embed-1b:free",
                                        input=query,
                                        encoding_format="float"
                                    )

    embedded_query = embedding.data[0].embedding
    most_similar_embeddings = await retrieve_most_similar(db, embedded_query)

    chunks = []

    for embedding in most_similar_embeddings:
        chunks.append(embedding.content)

    return chunks



    
    
    

    