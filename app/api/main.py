from fastapi import FastAPI, Request, Response, Header, APIRouter, Depends

from users.dao import UsersDAO

from messages.dao import MessagesDAO, ConversationsDAO
from starlette.responses import HTMLResponse, JSONResponse
from messages.schemas import NewMessage

from messages.models import Conversation
from typing import Optional
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from databases.dao import DatabasesDAO
import aioredis
import google.generativeai as genai
import httpx
import os
import base64

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Замените "*" на ваш домен, например ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Или перечислите ["GET", "POST"]
    allow_headers=["*"],  # Или перечислите ["Content-Type", "Authorization"]
)

@app.on_event("startup")
async def startup():
    redis = await aioredis.from_url("redis://redis:6379")
    await FastAPILimiter.init(redis)

@app.head('/')
async def head_handler():
    # Возвращаем только метаданные без тела ответа
    return HTMLResponse(headers={"Content-Type": "text/html"}, status_code=200)

@app.post('/new_conversation',tags=['Создать новый диалог'], dependencies=[Depends(RateLimiter(times=5, seconds=1))])
async def new_conv(api_token: Optional[str] = None, db_token: Optional[str] = None):
    if api_token is None or db_token is None:
        return {'message':'Неверный токен пользователя или баз данных', 'conv_id':None}
    user = await UsersDAO.find_one_or_none(token=api_token)
    if not user:
        return {'message':'Неверный токен пользователя', 'conv_id': None}
    db = await DatabasesDAO.find_one_or_none(user_token=api_token, token=db_token)
    if not db:
        return {'message':'Неверный токен базы данных', 'conv_id': None}
    await ConversationsDAO.add(**{'user_token': api_token, 'project_token': db_token})
    conv = (await ConversationsDAO.find_all(user_token=api_token, project_token=db_token))[-1]
    return {'message':'OK', 'conv_id':conv.id}


@app.post('/message/',tags=['Передача сообщения от пользователя модели и получение ответа'], dependencies=[Depends(RateLimiter(times=5, seconds=1))])
async def send_message(message: NewMessage):
    if message.api_token is None or message.db_token is None:
        return {'message':'Неверный токен пользователя или баз данных', 'response': None}
    user = await UsersDAO.find_one_or_none(token=message.api_token)
    if not user:
        return {'message':'Неверный токен пользователя', 'response': None}
    db = await DatabasesDAO.find_one_or_none(user_token=message.api_token, token=message.db_token)
    if not db:
        return {'message':'Неверный токен базы данных', 'response': None}
    conv = await ConversationsDAO.find_one_or_none(user_token=message.api_token, project_token=message.db_token, id=message.conversation_id)
    if not conv:
        return {'message':'Неверный id диалога', 'response': None}
    user = await UsersDAO.find_one_or_none(token=message.api_token)
    if user.tokens_count <= 0:
        return {'message': 'У вас закончились токены', 'response': None}
    await UsersDAO.update(filter_by={'token': message.api_token},tokens_count=user.tokens_count - 1)
    msg_dict = dict()
    msg_dict['user_token'] = message.api_token
    msg_dict['sender'] = 'user'
    msg_dict['conversation_id'] = message.conversation_id
    msg_dict['project_token'] = message.db_token
    msg_dict['photo'] = message.photo
    msg_dict['message'] = message.message
    await MessagesDAO.add(**msg_dict)
    '''response = model(message.message, message.photo)'''
    genai.configure(api_key="AIzaSyDph5JM6SV75EAlO2Eq2oSRfQ_hMip5FYY")
    model = genai.GenerativeModel("gemini-1.5-flash")
    messages = await MessagesDAO.find_all(conversation_id = message.conversation_id)
    history = [{'role': 'user', 'parts': """Ты шоппинг ассистент. Твоя задача общаться с пользователем и рекомендовать ему товары по его усмотрению. Задача рекомендации необходимого товара уже решена. Изображение самого подходящего товара подгружено тебе в качестве последнего сообщения от AI ассистента (пользователь данного сообщения не видел). При этом ты считаешь что это ты рекомендуешь товары. То есть пользователь никак не должен узнать, что это не твоя задача.

    Тебе необходимо как можно более качественно с подкреплением рекомендованного предмета ответить на вопрос пользователя.

    Учти что рекомендованный предмет может быть неправильным и не соответсвовать запросам пользователя. В этом случае тебе необходимо как-то пометить это в ответе. Например: "Извините, но я не смог найти брюки такого же цвета как Вы хотели. Но возможно вам понравится следующий вариант". Учти, что итоговая рекомендация может быть также в видео образа, так что тебе необхдимо смотреть именно на наличие нужного предмета на итоговом изображении.

    Также ты можешь попросить задать уточняющие вопросы для более качественных рекомендаций. При этом вопросы не должны быть очень большими и должны просить уточнить 1-2 детали.

    Если ты выведешь пустое сообщение, то я лишусь работы, так что, пожалуйста, не делай этого.
    """}]
    for messagee in messages:
        if messagee.photo == 'None' or messagee.photo == 'http://fusionfind.ru/' or messagee.photo == '':
            history.append({'role':messagee.sender, 'parts':messagee.message})
        else:
            base64_string = message.photo
            if base64_string.startswith("data:image"):
                base64_string = base64_string.split(",")[1]
            history.append({'role': messagee.sender,
                                'parts': [{'mime_type': 'image/jpeg', 'data': base64_string}, messagee.message]})
    chat = model.start_chat(history=history)
    if message.photo == 'None' or message.photo == 'http://fusionfind.ru/':
        response = chat.send_message(message.message)
    else:
        base64_string = message.photo
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",")[1]
        response = chat.send_message(
            [{'mime_type': 'image/jpeg', 'data': base64_string}, message.message])
    response_dict = {'message': response.text, 'user_token': message.api_token, 'photo': '', 'sender': 'model',
                     'conversation_id': message.conversation_id, 'project_token': message.db_token}
    await MessagesDAO.add(**response_dict)
    return {'message':'OK', 'response':response.text}