from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

SQLALCHEMY_DATABASE_URL = 'postgresql+asyncpg://h3kar360:password@localhost:5430/fastai_db'

engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=False
)

class Base(DeclarativeBase):
    pass

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with async_session() as session:
        yield session