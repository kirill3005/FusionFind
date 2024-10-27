from fastapi.security import APIKeyHeader
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta, timezone

from pydantic import EmailStr
from requests import Request

from config import get_auth_data, settings
from users.dao import UsersDAO

from fastapi import Request, HTTPException, status, Depends, Security

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def get_password_hash(password:str) -> str:
    return pwd_context.hash(password)

async def verify_password(plain_password:str, hashed_password:str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

async def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=30)
    to_encode.update({"exp": expire})
    auth_data = get_auth_data()
    encode_jwt = jwt.encode(to_encode, auth_data['secret_key'], algorithm=auth_data['algorithm'])
    return encode_jwt

async def authenticate_user(email: EmailStr, password: str):
    user = await UsersDAO.find_one_or_none(email=email)
    if not user or await verify_password(plain_password=password, hashed_password=user.password) is False:
        return None
    return user


async def check_access_token(
        request: Request,
        authorization_header: str = Security(APIKeyHeader(name='Authorization', auto_error=False))
) -> str:
    # Проверяем, что токен передан
    if authorization_header is None:
        raise {'message':'Ошибка при проверке токена'}

    # Убираем лишнее из токена
    clear_token = authorization_header.replace('Bearer ', '')

    try:
        # Проверяем валидность токена
        payload = jwt.decode(clear_token, key=settings.SECRET_KEY, algorithms=settings.ALGORITHM)
    except {'message':'Ошибка при проверке токена'}:
        # В случае невалидности возвращаем ошибку
        raise {'message':'Ошибка при проверке токена'}

    # Идентифицируем пользователя
    user = await UsersDAO.find_one_or_none(id=payload['sub'])
    if not user:
        raise {'message':'Ошибка при проверке токена'}

    request.state.user = user

    return authorization_header

import secrets
import base64

def generate_single_part_token(db_id: int):
    # Генерируем случайные байты
    random_bytes = secrets.token_bytes(16)
    # Создаем строку на основе id и случайных байт
    token_input = f"{db_id}-{base64.urlsafe_b64encode(random_bytes).decode('utf-8')}"
    # Кодируем строку в base64
    token = base64.urlsafe_b64encode(token_input.encode('utf-8')).decode('utf-8').rstrip("=")
    return token
