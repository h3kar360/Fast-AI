from fastapi import APIRouter, Depends, HTTPException, status, Response, UploadFile, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import json

from app.db import get_db
from app.api import crud
from app.api.schema import CreateChat, UpdateContext, ChatInfo, ChatInput, ChatResponse, CreateDocs, DocsCreatedResponse
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

    # merge all the contents
    content = f'{context}$/$user:{message}'
    messages = [{ "role": "user", "content": content }]

    # agent loop
    while True:
        llm = await client.chat.completions.create(
            model='nvidia/nemotron-3.5-lightning:free',
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Search Cat Document",
                        "description": "Searches internal knowledge base to retrieve relevant context about cats.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search terms or question to embed and look up in vector storage."
                                }
                            }
                        }
                    },
                    "required": ["query"]
                },
            ]
        )

        messages.append(llm.choices[0].message)

        if not llm.choices or len(llm.choices) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='No choices returned from LLM provider'
            )

        # stop when no tools left and llm is ready to respond
        if not llm.choices[0].message.tool_calls:
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

        # tool logic
        if llm.choices[0].message.tool_calls:
            for tool_call in llm.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                tool_call_id = tool_call.id
                function_arguments = json.loads(tool_call.function.arguments)

                if function_name == 'Search Cat Document':
                    query = function_arguments['query']
                    chunks = await crud.retrieve_relevant_chunks(db, query)  

                    str_chunks = "\n".join(f"- {chunk}" for chunk in chunks)
                                      
                    messages.append({ "role": "tool", "tool_call_id": tool_call_id, "content": str_chunks })

@router.post('/api/docs', response_model=DocsCreatedResponse)
async def add_docs(file: UploadFile, title: str = Form(), db: AsyncSession = Depends(get_db)):
    contents = await file.read()
    text = contents.decode('utf-8', errors='ignore')
    # create new doc
    doc_info = CreateDocs(title=title)
    doc = await crud.create_doc(db, doc_info)

    # get the doc
    dataset = [line.strip() for line in text.split('\n') if line.strip()]

    # add chunks to embeddings
    for i, chunk in enumerate(dataset):
        await crud.add_chunks(db, chunk, doc.id)

    return doc

                    
@router.delete('/api/chats/{chat_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: int, db: AsyncSession = Depends(get_db)):
    deleted = await crud.delete_chat(db, chat_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Chat with id {chat_id} not found"
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)