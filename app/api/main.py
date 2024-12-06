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
import requests
from random import randint

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Замените "*" на ваш домен, например ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Или перечислите ["GET", "POST"]
    allow_headers=["*"],  # Или перечислите ["Content-Type", "Authorization"]
)
df = ''
keys = []

ind = 0
geminies = []


@app.on_event("startup")
async def startup():
    redis = await aioredis.from_url("redis://redis:6379")
    await FastAPILimiter.init(redis)
    global model_emb
    model_emb = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
    global df
    tensors = torch.load('tensors.pt')
    df = pd.read_csv('dataset_with_captions.csv')
    df['items_embeddings'] = [torch.tensor(i) for i in tensors]
    keys = ['AIzaSyA04Pr0_oKqpuoLiTBLBD5QaTKofVp0qvE', 'AIzaSyBayHdeycAi1-S6dY--3YZRk3qY-4HRZBM',
            'AIzaSyBLFeVGiRFmlHBqhUU4dkr24P0GLRQpk5E', 'AIzaSyAMNZYp9mj8dDvQ0d0Qc24nMaVrp536Syw',
            'AIzaSyD2cA3Mduptwd_f-Xf2_gkkC6sMdYysECI', 'AIzaSyCljkTFey7phavWeXWh8LCMlI_Cm9KsDmo',
            'AIzaSyBm4Nzv6YGcX0bb9uL3wKTd4dbScxVAGY4', 'AIzaSyC0skPs3B7jMmnB6GZaiARrFfBjuUdC2bs']
    global geminies
    global ind
    ind = randint(0, 7)
    for i in range(8):
        genai.configure(api_key=keys[i])
        geminies.append(genai.GenerativeModel("gemini-1.5-flash"))


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


@app.post('/new_conversation', tags=['Создать новый диалог'], dependencies=[Depends(RateLimiter(times=5, seconds=1))])
async def new_conv(api_token: Optional[str] = None, db_token: Optional[str] = None):
    global is_downloaded
    if api_token is None or db_token is None:
        return {'message': 'Неверный токен пользователя или баз данных', 'conv_id': None}
    user = await UsersDAO.find_one_or_none(token=api_token)
    if not user:
        return {'message': 'Неверный токен пользователя', 'conv_id': None}
    db = await DatabasesDAO.find_one_or_none(user_token=api_token, token=db_token)
    if not db:
        return {'message': 'Неверный токен базы данных', 'conv_id': None}
    await ConversationsDAO.add(**{'user_token': api_token, 'project_token': db_token})
    conv = (await ConversationsDAO.find_all(user_token=api_token, project_token=db_token))[-1]
    return {'message': 'OK', 'conv_id': conv.id}


@app.post('/message/', tags=['Передача сообщения от пользователя модели и получение ответа'],
          dependencies=[Depends(RateLimiter(times=5, seconds=1))])
async def send_message(message: NewMessage):
    '''if message.api_token is None or message.db_token is None:
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
    await UsersDAO.update(filter_by={'token': message.api_token},tokens_count=user.tokens_count - 1)'''
    msg_dict = dict()
    msg_dict['user_token'] = message.api_token
    msg_dict['sender'] = 'user'
    msg_dict['conversation_id'] = int(message.conversation_id)
    msg_dict['project_token'] = message.db_token
    msg_dict['photo'] = message.photo
    msg_dict['message'] = message.message
    await MessagesDAO.add(**msg_dict)

    messages = await MessagesDAO.find_all(conversation_id=int(message.conversation_id))

    history = [{'role': 'user', 'parts': """system: Ты шоппинг ассистент. Твоя задача общаться с пользователем и рекомендовать ему товары по его усмотрению.

    Если надо найти предмет, то это делает другой сервис и нужный предмет уже найден и передаётся тебе как второе фото. Отвечай так, чтобы было понятно, что нужный предмет уже найден и передан пользователю.

    Учти что рекомендованный предмет может быть неправильным и не соответсвовать запросам пользователя. В этом случае тебе необходимо как-то пометить это в ответе. Например: "Извините, но я не смог найти брюки такого же цвета как Вы хотели. Но возможно вам понравится следующий вариант". Учти, что итоговая рекомендация может быть также в видео образа, так что тебе необхдимо смотреть именно на наличие нужного предмета на итоговом изображении.

    Также ты можешь попросить задать уточняющие вопросы для более качественных рекомендаций. При этом вопросы не должны быть очень большими и должны просить уточнить 1-2 детали. Поддерживай диалог с пользователем, отвечай на его вопросы.

    Если ты выведешь пустое сообщение, то я лишусь работы, так что, пожалуйста, не делай этого.
    """}]
    for messagee in messages:
        if messagee.photo == 'None' or messagee.photo == 'http://fusionfind.ru/' or messagee.photo == '':
            history.append({'role': messagee.sender, 'parts': messagee.message})
        else:
            base64_string4 = messagee.photo
            if base64_string4.startswith("data:image"):
                base64_string = base64_string4.split(",")[1]
                img_type = base64_string4.split(",")[0].split(';')[0].replace('data:', '')
            else:
                base64_string = base64.b64encode(requests.get(base64_string4).content).decode("utf-8")
                img_type = f'image/{messagee.photo.split(".")[-1]}'
            history.append({'role': messagee.sender,
                            'parts': [{'mime_type': img_type, 'data': base64_string}, messagee.message]})

    # history.append({'role':'user', 'parts': [{'mime_type': img_type, 'data': base64_string}]})
    global ind
    chat = geminies[ind % 8].start_chat(history=history)
    if message.photo == 'None' or message.photo == 'http://fusionfind.ru/':
        response = chat.send_message(message.message)
        classify = geminies[ind % 8].generate_content(
            f'system: Ты ассистент, классифицирующий входные сообщения от пользователей шопинг ассистенту на те, после которых надо вызвать поиск по базе товаров и те, после которых не надо вызвать поиск. Выведи только 1, если надо и только 0, если не надо. Не надо вызывать поиск по вопросу о товаре, если это не требуется. Определи класс следующего сообщения: {message.message}').text[
            0]
    else:
        base64_string2 = message.photo
        if base64_string2.startswith("data:image"):
            base64_string = base64_string2.split(",")[1]
            img_type = base64_string2.split(",")[0].split(';')[0].replace('data:', '')
        else:
            base64_string = base64.b64encode(requests.get(base64_string).content).decode("utf-8")
            img_type = f'image/{messagee.photo.split(".")[-1]}'
        classify = geminies[ind % 8].generate_content([{'mime_type': img_type, 'data': base64_string},
                                                       f'system: Ты ассистент, классифицирующий входные сообщения от пользователей шопинг ассистенту на те, после которых надо вызвать поиск по базе товаров и те, после которых не надо вызвать поиск. Не надо вызывать поиск по вопросу о товаре, если это не требуется и не всегда отправка изображения значит, что надо делать поиск. Выведи только 1, если надо и только 0, если не надо, без лишнего. Например: найди мне эту рубашку - "1". Определи класс следующего сообщения с фото, отправленного с ним: {message.message}']).text[
            0]
    if classify == '1':
        try:
            caption = geminies[ind % 8].generate_content([{'mime_type': img_type, 'data': base64_string}, f"""Ты ассистент для генерации описаний товаров. На вход тебе подаётся изображение и текст запроса от пользователя. Твоя задача — создать максимально точное описание того товара на изображении, который укажет пользователь. Ты должен писать только факты.

                Пользователь написал: {message.message}.

                Сгенерируй максимально полное описание, включая следующие характеристики:
                1. Материал 
                2. Цвет 
                3. Дополнительные детали (максимально подробно, не менее 30 слов)
                4. Бренд (если видно, если бренда нет, то указать "не возможно распознать")
                5. Примерную стоимость (если не можешь определить, то предположи в виде разброса. Пример: 1500-2000. Разброс не должен превышать 40% от стоимости товара)

                Твой ответ должен содержать все ключевые детали для точного поиска."""])
            caption = caption.text
            item_image, item_name = make_recommendation(caption, df)
            base64_string3 = base64.b64encode(requests.get(item_image, stream=True, timeout=30).content).decode("utf-8")
            img_type2 = f'image/' + item_image.split('.')[-1]
            response = chat.send_message(
                [{'mime_type': img_type, 'data': base64_string}, {'mime_type': img_type2, 'data': base64_string3},
                 message.message])
        except:
            caption = geminies[ind % 8].generate_content(f"""Ты ассистент для генерации описаний товаров. На вход тебе подаётся текст запроса от пользователя. Твоя задача — создать максимально точное описание того товара на изображении, который укажет пользователь. Ты должен писать только факты.

            Пользователь написал: {message.message}.

            Сгенерируй максимально полное описание, включая следующие характеристики:
            1. Материал 
            2. Цвет 
            3. Дополнительные детали (максимально подробно, не менее 30 слов)
            4. Бренд (если видно, если бренда нет, то указать "не возможно распознать")
            5. Примерную стоимость (если не можешь определить, то предположи в виде разброса. Пример: 1500-2000. Разброс не должен превышать 40% от стоимости товара)

            Твой ответ должен содержать все ключевые детали для точного поиска.""")
            caption = caption.text
            item_image, item_name = make_recommendation(caption, df)
            base64_string3 = base64.b64encode(requests.get(item_image, stream=True, timeout=30).content).decode("utf-8")
            img_type2 = f'image/' + item_image.split('.')[-1]
            response = chat.send_message(
                [{'mime_type': img_type2, 'data': base64_string3}, message.message])
    else:
        try:
            response = chat.send_message(
                [{'mime_type': img_type, 'data': base64_string}, message.message])
        except:
            response = chat.send_message(
                message.message)
        item_image = ''
    response_dict = {'message': response.text, 'user_token': message.api_token, 'photo': item_image, 'sender': 'model',
                     'conversation_id': int(message.conversation_id), 'project_token': message.db_token}
    await MessagesDAO.add(**response_dict)
    ind += 1
    return {'message': 'OK', 'response': [response.text, item_image]}
