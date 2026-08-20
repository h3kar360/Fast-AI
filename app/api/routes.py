from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.db import get_db
from app.api import crud
from app.api.schema import CreateChat, UpdateContext, ChatInfo, ChatInput, ChatResponse
from app.config import client

router = APIRouter()

@router.get('/api/chats', response_model=List[ChatInfo])
async def get_all_chats(db: AsyncSession = Depends(get_db)):
    return await crud.get_all_chats(db)

@router.get('/api/chats/{chat_id}', response_model=ChatInfo)
async def get_chat(chat_id: int, db: AsyncSession = Depends(get_db)):
    chat = await crud.get_chat_by_id(db, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f'Chat with id {chat_id} not found'
        )
    return chat

@router.post('/api/chats', response_model=ChatInfo, status_code=status.HTTP_201_CREATED)
async def create_chat(new_chat: CreateChat, db: AsyncSession = Depends(get_db)):
    return await crud.create_chat(db, new_chat)
    
@router.post('/api/chats/{chat_id}', response_model=ChatResponse)
async def chat_to_llm(chat_id: int, chat_message: ChatInput, db: AsyncSession = Depends(get_db)):
    chat = await crud.get_chat_by_id(db, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f'Chat with id {chat_id} not found'
        )

    context = chat.chat_context or ""
    message = chat_message.message

    content = f'{context}$/$user:{message}'

    llm = await client.chat.completions.create(
        model='nvidia/nemotron-3.5-lightning:free',
        messages=[{ "role": "user", "content": content }]
    )

    if not llm.choices or len(llm.choices) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='No choices returned from LLM provider'
        )

    response = llm.choices[0].message.content or ""

    updated_context = content + f'$/$assistant:{response}'

    updated_chat = await crud.update_context(db, updated_context, chat_id)

    if not updated_chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Failed to update chat context"
        )

    return ChatResponse(
        id=updated_chat.id,
        title=updated_chat.title,
        response=response
    )

@router.delete('/api/chats/{chat_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: int, db: AsyncSession = Depends(get_db)):
    deleted = await crud.delete_chat(db, chat_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Chat with id {chat_id} not found"
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)

