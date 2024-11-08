from fastapi import FastAPI, Request, Response, Header, APIRouter, Depends

from users.dao import UsersDAO

from messages.dao import MessagesDAO, ConversationsDAO

from messages.schemas import NewMessage

from messages.models import Conversation
from typing import Optional
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from databases.dao import DatabasesDAO
import aioredis

app = FastAPI()

@app.on_event("startup")
async def startup():
    redis = await aioredis.from_url("redis://redis:6379")
    await FastAPILimiter.init(redis)
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
async def send_message(message: NewMessage, api_token: Optional[str] = None, db_token: Optional[str] = None, conversation_id: Optional[int]=None):
    if api_token is None or db_token is None:
        return {'message':'Неверный токен пользователя или баз данных', 'response': None}
    user = await UsersDAO.find_one_or_none(token=api_token)
    if not user:
        return {'message':'Неверный токен пользователя', 'response': None}
    db = await DatabasesDAO.find_one_or_none(user_token=api_token, token=db_token)
    if not db:
        return {'message':'Неверный токен базы данных', 'response': None}
    conv = await ConversationsDAO.find_one_or_none(user_token=api_token, project_token=db_token, id=conversation_id)
    if not conv:
        return {'message':'Неверный id диалога', 'response': None}
    user = await UsersDAO.find_one_or_none(token=api_token)
    if user.tokens_count <= 0:
        return {'message': 'У вас закончились токены', 'response': None}
    await UsersDAO.update(filter_by={'token': api_token},tokens_count=user.tokens_count - 1)
    msg_dict = message.dict()
    msg_dict['user_token'] = api_token
    msg_dict['sender'] = 'user'
    msg_dict['conversation_id'] = conversation_id
    msg_dict['project_token'] = db_token
    await MessagesDAO.add(**msg_dict)
    '''response = model(message.message, message.photo)'''
    response_dict = {'message': 'response', 'user_token': api_token, 'photo': '', 'sender': 'model',
                     'conversation_id': conversation_id, 'project_token': db_token}
    await MessagesDAO.add(**response_dict)
    return {'message':'OK', 'response':'Ответ от модели'}