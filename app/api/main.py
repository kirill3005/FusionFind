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
from sentence_transformers import SentenceTransformer

from fastapi.middleware.cors import CORSMiddleware
import torch
from torch.nn.functional import cosine_similarity
import pandas as pd


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
    global model_emb
    model_emb = SentenceTransformer('intfloat/multilingual-e5-large-instruct')


is_downloaded = False
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def get_embedding(caption):
    embeddings = torch.tensor(model_emb.encode(
        get_detailed_instruct("Получив описание товара, найдите соответствующие товары в базе данных", caption),
        normalize_embeddings=True))
    return embeddings


def find_closest_row(caption, dataset, column_name='items_embeddings'):
    """
    Находит строку в DataFrame с минимальным косинусным расстоянием.

    Args:
        caption (torch.Tensor): Исходный тензор.
        dataset (pd.DataFrame): DataFrame с эмбеддингами в указанной колонке.
        column_name (str): Название колонки с эмбеддингами.

    Returns:
        pd.Series: Строка DataFrame с минимальным косинусным расстоянием.
    """
    caption = get_embedding(caption)
    similarities = dataset[column_name].apply(lambda x: cosine_similarity(caption.unsqueeze(0), x.unsqueeze(0)).item())

    closest_index = similarities.idxmax()

    return dataset.loc[closest_index]


def make_recommendation(caption, dataset):

    item = find_closest_row(caption, dataset)

    return item['items_images'], item['image_name']

@app.head('/')
async def head_handler():
    # Возвращаем только метаданные без тела ответа
    return HTMLResponse(headers={"Content-Type": "text/html"}, status_code=200)

@app.post('/new_conversation',tags=['Создать новый диалог'], dependencies=[Depends(RateLimiter(times=5, seconds=1))])
async def new_conv(api_token: Optional[str] = None, db_token: Optional[str] = None):
    global is_downloaded
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
    conv = await ConversationsDAO.find_one_or_none(user_token=message.api_token, project_token=message.db_token, id=int(message.conversation_id))
    if not conv:
        return {'message':'Неверный id диалога', 'response': None}
    user = await UsersDAO.find_one_or_none(token=message.api_token)
    if user.tokens_count <= 0:
        return {'message': 'У вас закончились токены', 'response': None}
    await UsersDAO.update(filter_by={'token': message.api_token},tokens_count=user.tokens_count - 1)
    msg_dict = dict()
    msg_dict['user_token'] = message.api_token
    msg_dict['sender'] = 'user'
    msg_dict['conversation_id'] = int(message.conversation_id)
    msg_dict['project_token'] = message.db_token
    msg_dict['photo'] = message.photo
    msg_dict['message'] = message.message
    await MessagesDAO.add(**msg_dict)
    '''response = model(message.message, message.photo)'''
    genai.configure(api_key="AIzaSyDph5JM6SV75EAlO2Eq2oSRfQ_hMip5FYY")
    model = genai.GenerativeModel("gemini-1.5-flash")
    messages = await MessagesDAO.find_all(conversation_id = int(message.conversation_id))
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
        caption = model.generate_content([{'mime_type': 'image/webp', 'data': base64_string}, f"""Ты ассистент для генерации описаний товаров. На вход тебе подаётся изображение и текст запроса от пользователя. Твоя задача — создать максимально точное описание того товара на изображении, который укажет пользователь. Ты должен писать только факты.

            Пользователь написал: {message.message}.

            Сгенерируй максимально полное описание, включая следующие характеристики:
            1. Материал 
            2. Цвет 
            3. Дополнительные детали (максимально подробно, не менее 30 слов)
            4. Бренд (если видно, если бренда нет, то указать "не возможно распознать")
            5. Примерную стоимость (если не можешь определить, то предположи в виде разброса. Пример: 1500-2000. Разброс не должен превышать 40% от стоимости товара)

            Твой ответ должен содержать все ключевые детали для точного поиска."""])
        df = pd.read_csv('dataset_with_captions.csv')
        item_image, item_name = make_recommendation(caption, df)
        base64_string1 = base64.b64encode(httpx.get(item_image.content).decode("utf-8"))
        if base64_string1.startswith("data:image"):
            base64_string1 = base64_string1.split(",")[1]
        response = chat.send_message(
            [{'mime_type': 'image/webp', 'data': base64_string}, {'mime_type': 'image/webp', 'data': base64_string1}, message.message+'Второе фото это фото найденного товара в базе данных. Ответь пользователю, используя информацию.'])
    response_dict = {'message': response.text, 'user_token': message.api_token, 'photo': '', 'sender': 'model',
                     'conversation_id': int(message.conversation_id), 'project_token': message.db_token}
    await MessagesDAO.add(**response_dict)
    return {'message':'OK', 'response':response.text}